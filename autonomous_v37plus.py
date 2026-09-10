"""Crash-resumable V37+ research controller.

Research outputs are built outside cert/, verified in staging, and committed by
atomic directory rename.  Seal opening is deliberately delegated to a separate
one-shot worker; this controller never reads a V36+ seal outcome while doing
research.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

import runtime_limits


ROOT=Path(__file__).resolve().parent
STATE_DIR=ROOT/"state"
RESEARCH_DIR=ROOT/"research"
STAGING_DIR=RESEARCH_DIR/"staging"
STATE_PATH=STATE_DIR/"AUTONOMOUS_LOOP_STATE.json"
JOURNAL_PATH=STATE_DIR/"AUTONOMOUS_LOOP_JOURNAL.jsonl"
LOCK_PATH=STATE_DIR/"AUTONOMOUS_CONTROLLER.lock"
LEDGER_JSONL=RESEARCH_DIR/"AUTONOMOUS_EXPERIMENT_LEDGER.jsonl"
LEDGER_CSV=RESEARCH_DIR/"AUTONOMOUS_EXPERIMENT_LEDGER.csv"
FINAL_MARKER=STATE_DIR/"FINAL_PASS"
REGISTRY_PATH=RESEARCH_DIR/"EXPERIMENT_REGISTRY.json"
CHAMPION_PATH=RESEARCH_DIR/"CURRENT_CHAMPION.json"
SOURCE_ACQUISITION_STATUS_PATH=RESEARCH_DIR/"SOURCE_ACQUISITION_STATUS.json"
EXPECTED_DEV_SHA="2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
NEW_EXACT_DEV_EXTENSION_FAMILY="NEW_EXACT_DEV_EXTENSION_RETRAIN"
DATA_EPOCH_CONTRACT_PATH=RESEARCH_DIR/"NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json"
DATA_EPOCH_ASSETS={
    "extension_csv":ROOT/"data"/"V59_DEV_EXTENSION_LABELED.csv.gz",
    "extension_manifest":ROOT/"data"/"V59_DEV_EXTENSION_MANIFEST.json",
    "role_assignment_json":ROOT/"data"/"V59_DATA_ROLE_ASSIGNMENT.json",
    "role_assignment_parquet":ROOT/"data"/"V59_DATA_ROLE_ASSIGNMENT.parquet",
}
DATA_EPOCH_MINIMUM={
    "total":240,
    "required_source_families":{"US_SEC":120,"KR_NEWS":120},
    "per_source":{"y_0":25,"y_1":25,"unique_dates":20,"unique_tickers":30},
    "chronological_blocks":{"count":5,"minimum_n":20,"require_both_classes":True},
}
DATA_EPOCH_EXACT_CONTRACT={
    "name":"EXACT_T2_30M_V36",
    "role":"DEV_EXTENSION",
    "price_cadence_sec":60,
    "bar_timestamp_semantics":"OPEN",
    "embargo_minutes":35,
    "reserved_roles_forbidden":["RESEARCH_SEAL_POOL","FINAL_META_RESERVE"],
}
REQUIRED_OUTPUTS=(
    "VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json",
    "MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json",
    "DATA_STATUS.json","SOURCE_POOL_STATUS.json","VERSION_DELTA.json","NEXT_ACTION.json",
    "RUN_STATUS.json","ARTIFACT_MANIFEST.json",
)

# A version number is consumed only when its materially different runner exists.
LEGACY_RESEARCH_PROGRAM=(
    (37,"TEXT_SIGNAL_SOURCE_SPECIFIC","experiment_v37_text.py",
     "Fold-fitted word/character/semantic text experts, source experts, and numeric late fusion."),
    (38,"OPPORTUNITY_MAGNITUDE","experiment_v38_opportunity.py",
     "Separate direction from material-move opportunity and high-confidence selection."),
    (39,"SEC_STRUCTURED_SEMANTICS","experiment_v39_sec_semantics.py",
     "SEC item/form/EX-99 semantic architecture with missing-body fallback."),
    (40,"CAUSAL_EVENT_PRIOR","experiment_v40_causal_prior.py",
     "Inner-fitted event taxonomy priors with shrinkage and no issuer memorization."),
    (41,"SECTOR_RELATIVE_STATE","experiment_v41_sector_relative.py",
     "Point-in-time sector-relative pre-event state and market regime features."),
    (42,"ROBUST_STACKED_ENSEMBLE","experiment_v42_robust_ensemble.py",
     "Chronological inner-OOF robust stacking across materially distinct families."),
    (43,"SOURCE_REGIME_EXPERT","experiment_v43_source_regime.py",
     "Time-decayed source-specific numeric experts with pooled shrinkage fallback."),
    (44,"MICROSTRUCTURE_SIGNATURE","experiment_v44_microstructure.py",
     "Completed-bar clock-window volatility, jump, range, signed-volume, and reversal signatures."),
    (45,"SIGNED_RETURN_REGRESSION","experiment_v45_signed_return.py",
     "Robust signed-return and absolute-magnitude regression fused with direction classification."),
    (46,"ARTICLE_CROSS_SECTION","experiment_v46_article_cross_section.py",
     "Outcome-free within-article cross-sectional ranks and deviations of pre-entry market state."),
    (47,"CROSS_MARKET_TRANSFER","experiment_v47_cross_market.py",
     "Past-US shared representation blended with local KR experts under market-balanced weights."),
    (48,"EVENT_FORM_XGB_EXPERT","experiment_v48_event_form_xgb.py",
     "GPU XGBoost pooled model plus shrinkage form/source and event-taxonomy experts."),
    (49,"STABLE_RANK_LOGIT","experiment_v49_stable_rank.py",
     "Empirical-rank sparse logistic using only features with stable chronological effect signs."),
    (50,"FROZEN_MULTILINGUAL_EMBEDDING","experiment_v50_embeddings.py",
     "Frozen multilingual E5 headline/semantic embeddings with fold-fitted linear late fusion."),
    (51,"PREQUENTIAL_SOURCE_ROUTING","experiment_v51_source_routing.py",
     "Past-OOF-only routing between economic microstructure and multilingual semantic experts by source."),
    (52,"NONLINEAR_EMBEDDING_HEAD","experiment_v52_embedding_mlp.py",
     "Fold-fitted GPU MLP over frozen headline/semantic embeddings with numeric late fusion."),
    (53,"DECOUPLED_DIRECTION_OPPORTUNITY","experiment_v53_decoupled_policy.py",
     "Fixed numeric direction model with independently fold-fitted V46 opportunity confidence."),
    (54,"DIVERSE_DIRECTION_CONSENSUS","experiment_v54_diverse_consensus.py",
     "Fixed equal-weight consensus of numeric, opportunity, and event-form direction experts with independently nested economic confidence."),
    (55,"RAW_BAR_TEMPORAL_CNN","experiment_v55_temporal_cnn.py",
     "GPU temporal residual CNN over the causal raw 120-minute one-minute-bar sequence, fixed-blended with the diverse consensus."),
    (56,"CAUSAL_SEMANTIC_RETRIEVAL","experiment_v56_semantic_retrieval.py",
     "Past-only multilingual semantic nearest-event retrieval for the distinct US news DGP with cluster exclusion and consensus fallback."),
    (57,"SUPERVISED_SEC_LANGUAGE_ADAPTATION","experiment_v57_sec_transformer.py",
     "Fold-specific GPU adaptation of the final multilingual encoder layers on structured SEC text, routed only to US SEC with fixed consensus blending."),
    (58,"LOCKED_SOURCE_DGP_ROUTING","experiment_v58_locked_source_routing.py",
     "Source-locked routing of numeric SEC, multilingual news, and diverse KR experts with one independent economic-confidence policy."),
)

DEFAULT_DYNAMIC_JOBS=(
    {"job_id":"US_SEC_STRUCTURED_SEMANTICS_V1","hypothesis":"US_SEC_STRUCTURED_SEMANTICS_V1",
     "family":"SEC_STRUCTURED_SEMANTICS","runner":"experiment_v59_sec_structured_v1.py",
     "description":"Segment-aware SEC text and explicit item, exhibit, taxonomy, state, and quantitative semantic experts routed into the locked V58 champion.",
     "priority":100,"status":"READY","assigned_version":59,"attempt_count":0},
)


def now()->str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical(value:Any)->bytes:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),
                      default=str).encode("utf-8")


def digest_bytes(value:bytes)->str:
    return hashlib.sha256(value).hexdigest()


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(payload,handle,ensure_ascii=False,indent=2,default=str)
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def read_json(path:Path,default:Any=None)->Any:
    if not path.exists():return default
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_research_gate(selected:dict[str,Any])->dict[str,Any]:
    """Recompute the immutable 14 research/seal gates from report metrics.

    Material hypothesis checks are deliberately excluded.  A runner may use
    them to select a challenger, but it cannot substitute them for this gate.
    """
    metrics=selected["metrics"];robust=selected["robustness"]
    bootstrap=robust["bootstrap"];markets=metrics["by_market"]
    folds=metrics["by_fold"];sources=metrics["by_source_family"]
    checks={
        "balanced_accuracy_ge_0_540":float(metrics["balanced_accuracy"])>=.540,
        "auc_ge_0_560":float(metrics["auc"])>=.560,
        "edge_ge_0_020":float(metrics["edge_vs_naive"])>=.020,
        "highconf_coverage_ge_0_15":float(metrics["highconf_coverage"])>=.15,
        "highconf_accuracy_ge_0_600":float(metrics["highconf_accuracy"])>=.600,
        "highconf_net_gt_0":float(metrics["strategy_mean_signed_net"])>0,
        "bootstrap_ba_lower_ge_0_500":float(bootstrap["balanced_accuracy_lower95"])>=.500,
        "bootstrap_highconf_net_lower_gt_0":float(bootstrap["highconf_strategy_net_lower95"])>0,
        "US_ba_ge_0_510":float(markets.get("US",{}).get("balanced_accuracy",0))>=.510,
        "KR_ba_ge_0_510":float(markets.get("KR",{}).get("balanced_accuracy",0))>=.510,
        "worst_fold_ba_ge_0_490":min(float(row["balanced_accuracy"]) for row in folds.values())>=.490,
        "no_major_source_ba_below_0_490":min(float(row["balanced_accuracy"]) for row in sources.values())>=.490,
        "winsorized_net_gt_0":float(robust["winsorized_net"])>0,
        "top_5_removed_net_gt_0":float(robust["top_5_removed_net"])>0,
    }
    return {"passed":sum(checks.values()),"total":len(checks),"checks":checks,
            "robust_survivor":all(checks.values())}


def is_new_exact_dev_extension_family(hypothesis:str)->bool:
    """Keep the V37-V68 identifier contract byte-for-byte on its legacy path."""
    return str(hypothesis).upper().startswith(NEW_EXACT_DEV_EXTENSION_FAMILY)


def extension_asset_metadata()->dict[str,dict[str,Any]]:
    metadata={}
    for name,path in DATA_EPOCH_ASSETS.items():
        if not path.is_file():continue
        metadata[name]={"path":str(path.relative_to(ROOT)),"sha256":sha256(path),
                        "bytes":path.stat().st_size}
    return metadata


def inspect_extension_support(path:Path)->tuple[dict[str,Any],dict[str,Any],list[str]]:
    """Audit retrain support without opening any reserved-pool label file."""
    required_sources=DATA_EPOCH_MINIMUM["required_source_families"]
    records={source:[] for source in required_sources};observed_sources=set();issues=[]
    counts={"total":0,"by_source_family":{}}
    if not path.is_file():return counts,{},[f"missing asset: {path.relative_to(ROOT)}"]
    required=("event_id","event_time_utc","ticker","market","source_family","y","data_role",
              "contract_version","price_cadence_sec","bar_timestamp_semantics")
    try:
        with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as handle:
            rows=csv.reader(handle);header=next(rows,None)
            if not header:return counts,{},["DEV_EXTENSION CSV has no header"]
            missing=[name for name in required if name not in header]
            if missing:return counts,{},[f"DEV_EXTENSION CSV missing columns: {missing}"]
            index={name:header.index(name) for name in required}
            for line_number,row in enumerate(rows,start=2):
                if len(row)!=len(header):
                    issues.append(f"DEV_EXTENSION malformed CSV row {line_number}");continue
                market=row[index["market"]].strip().upper()
                source=row[index["source_family"]].strip().upper()
                role=row[index["data_role"]].strip().upper()
                contract=row[index["contract_version"]].strip()
                cadence=row[index["price_cadence_sec"]].strip()
                semantics=row[index["bar_timestamp_semantics"]].strip().upper()
                if role!="DEV_EXTENSION":
                    qualifier="reserved" if role in DATA_EPOCH_EXACT_CONTRACT["reserved_roles_forbidden"] else "forbidden"
                    issues.append(f"{qualifier} role in extension CSV row {line_number}: {role}")
                observed_sources.add(source)
                if source not in required_sources:
                    issues.append(f"forbidden/reserved source_family in extension CSV row {line_number}: {source}")
                expected_market={"US_SEC":"US","KR_NEWS":"KR"}.get(source)
                if expected_market is not None and market!=expected_market:
                    issues.append(f"source/market mismatch in extension CSV row {line_number}: {source}/{market}")
                if contract!=DATA_EPOCH_EXACT_CONTRACT["name"]:
                    issues.append(f"contract mismatch in extension CSV row {line_number}: {contract}")
                if cadence!=str(DATA_EPOCH_EXACT_CONTRACT["price_cadence_sec"]):
                    issues.append(f"cadence mismatch in extension CSV row {line_number}: {cadence}")
                if semantics!=DATA_EPOCH_EXACT_CONTRACT["bar_timestamp_semantics"]:
                    issues.append(f"bar semantics mismatch in extension CSV row {line_number}: {semantics}")
                counts["total"]+=1
                if source not in required_sources:continue
                label=row[index["y"]].strip()
                if label not in {"0","1"}:
                    issues.append(f"non-binary y in extension CSV row {line_number}: {label}");continue
                timestamp_text=row[index["event_time_utc"]].strip()
                try:
                    timestamp=datetime.fromisoformat(timestamp_text.replace("Z","+00:00"))
                    if timestamp.tzinfo is None:raise ValueError("timezone missing")
                    timestamp=timestamp.astimezone(timezone.utc)
                except ValueError:
                    issues.append(f"invalid event_time_utc in extension CSV row {line_number}: {timestamp_text}")
                    continue
                ticker=row[index["ticker"]].strip().upper()
                if not ticker:
                    issues.append(f"empty ticker in extension CSV row {line_number}");continue
                records[source].append((timestamp,row[index["event_id"]].strip(),ticker,int(label)))
    except (OSError,EOFError,csv.Error,UnicodeError) as error:
        issues.append(f"DEV_EXTENSION CSV unreadable: {error!r}")

    block_rule=DATA_EPOCH_MINIMUM["chronological_blocks"]
    source_checks={}
    for source,minimum_n in required_sources.items():
        ordered=sorted(records[source],key=lambda row:(row[0],row[1]))
        labels=[row[3] for row in ordered]
        blocks=[];block_checks=[];block_count=int(block_rule["count"])
        for block in range(block_count):
            start=(block*len(ordered))//block_count
            stop=((block+1)*len(ordered))//block_count
            part=ordered[start:stop];part_labels=[row[3] for row in part]
            block_payload={"block":block+1,"n":len(part),
                "y_0":part_labels.count(0),"y_1":part_labels.count(1),
                "both_classes":set(part_labels)=={0,1},
                "start_utc":part[0][0].isoformat() if part else None,
                "end_utc":part[-1][0].isoformat() if part else None}
            blocks.append(block_payload)
            block_checks.append(block_payload["n"]>=int(block_rule["minimum_n"]) and
                                (block_payload["both_classes"] or
                                 not block_rule["require_both_classes"]))
        source_counts={"n":len(ordered),"y_0":labels.count(0),"y_1":labels.count(1),
            "unique_dates":len({row[0].date() for row in ordered}),
            "unique_tickers":len({row[2] for row in ordered}),
            "chronological_blocks":blocks}
        counts["by_source_family"][source]=source_counts
        per_source=DATA_EPOCH_MINIMUM["per_source"]
        source_checks[source]={
            "n_ge_minimum":source_counts["n"]>=int(minimum_n),
            "y_0_ge_minimum":source_counts["y_0"]>=int(per_source["y_0"]),
            "y_1_ge_minimum":source_counts["y_1"]>=int(per_source["y_1"]),
            "unique_dates_ge_minimum":source_counts["unique_dates"]>=int(per_source["unique_dates"]),
            "unique_tickers_ge_minimum":source_counts["unique_tickers"]>=int(per_source["unique_tickers"]),
            "chronological_5_blocks_each_n_ge_20_and_both_classes":all(block_checks),
        }
    support={
        "total_n_ge_240":counts["total"]>=int(DATA_EPOCH_MINIMUM["total"]),
        "only_required_source_families":observed_sources==set(required_sources),
        "by_source_family":source_checks,
    }
    return counts,support,issues


def inspect_data_epoch_contract()->dict[str,Any]:
    """Return a label-role-safe, fail-closed snapshot for new-data retraining."""
    issues=[]
    before=extension_asset_metadata()
    missing_assets=sorted(set(DATA_EPOCH_ASSETS)-set(before))
    issues.extend(f"missing asset: {name}" for name in missing_assets)
    counts,support,row_issues=inspect_extension_support(DATA_EPOCH_ASSETS["extension_csv"])
    issues.extend(row_issues)

    manifest=read_json(DATA_EPOCH_ASSETS["extension_manifest"],{}) or {}
    roles=read_json(DATA_EPOCH_ASSETS["role_assignment_json"],{}) or {}
    if manifest:
        if manifest.get("role")!="DEV_EXTENSION":issues.append("extension manifest role is not DEV_EXTENSION")
        if manifest.get("selection_uses_labels") is not False:
            issues.append("extension manifest selection_uses_labels must be false")
        if manifest.get("labels_opened_after_role_freeze") is not True:
            issues.append("extension labels were not opened after role freeze")
        manifest_counts={"US_SEC":int(manifest.get("US",-1)),
                         "KR_NEWS":int(manifest.get("KR",-1)),
                         "total":int(manifest.get("rows",-1))}
        csv_manifest_counts={
            "US_SEC":int(counts["by_source_family"].get("US_SEC",{}).get("n",-1)),
            "KR_NEWS":int(counts["by_source_family"].get("KR_NEWS",{}).get("n",-1)),
            "total":int(counts["total"]),
        }
        if manifest_counts!=csv_manifest_counts:
            issues.append("extension manifest source-family counts differ from CSV")
        csv_meta=before.get("extension_csv",{})
        if manifest.get("output_sha256")!=csv_meta.get("sha256"):
            issues.append("extension manifest output_sha256 differs from CSV")
        role_meta=before.get("role_assignment_parquet",{})
        if manifest.get("role_assignment_sha256")!=role_meta.get("sha256"):
            issues.append("extension manifest role-assignment SHA differs from parquet")
    else:issues.append("extension manifest is missing or invalid")
    if roles:
        if roles.get("selection_uses_labels") is not False:
            issues.append("role assignment selection_uses_labels must be false")
        if roles.get("labels_opened") is not False:
            issues.append("role assignment must precede label opening")
    else:issues.append("role assignment JSON is missing or invalid")

    after=extension_asset_metadata()
    if before!=after:issues.append("data-epoch assets changed during controller audit")
    assets=after
    support_pass=(support.get("total_n_ge_240") is True and
                  support.get("only_required_source_families") is True and
                  all(all(checks.values()) for checks in
                      support.get("by_source_family",{}).values()) and
                  set(support.get("by_source_family",{}))==
                      set(DATA_EPOCH_MINIMUM["required_source_families"]))
    if not support_pass:
        issues.append(f"DEV retrain support contract not met: {support}")

    contract=read_json(DATA_EPOCH_CONTRACT_PATH)
    contract_sha=sha256(DATA_EPOCH_CONTRACT_PATH) if DATA_EPOCH_CONTRACT_PATH.is_file() else None
    if not isinstance(contract,dict):
        issues.append(f"active freeze contract absent: {DATA_EPOCH_CONTRACT_PATH.relative_to(ROOT)}")
        contract={}
    else:
        if contract.get("schema_version")!=1:issues.append("freeze contract schema_version must be 1")
        if contract.get("family")!=NEW_EXACT_DEV_EXTENSION_FAMILY:
            issues.append("freeze contract family mismatch")
        if contract.get("status")!="FROZEN" or contract.get("frozen") is not True:
            issues.append("freeze contract is not FROZEN")
        freeze_id=str(contract.get("freeze_id","")).strip()
        if len(freeze_id)<12:issues.append("freeze_id is missing or too short")
        if not str(contract.get("frozen_at","")).strip():issues.append("frozen_at is missing")
        if contract.get("selection_uses_labels") is not False:
            issues.append("freeze selection_uses_labels must be false")
        if contract.get("role")!="DEV_EXTENSION":issues.append("freeze role must be DEV_EXTENSION")
        if contract.get("minimum_support")!=DATA_EPOCH_MINIMUM:
            issues.append("freeze minimum_support differs from controller minimum")
        if contract.get("counts")!=counts:issues.append("freeze counts differ from extension CSV")
        if contract.get("data_contract")!=DATA_EPOCH_EXACT_CONTRACT:
            issues.append("freeze exact-data contract mismatch")
        declared=contract.get("files",{})
        if not isinstance(declared,dict):
            issues.append("freeze files declaration is invalid");declared={}
        for name,metadata in assets.items():
            if declared.get(name)!=metadata:issues.append(f"freeze asset declaration mismatch: {name}")
        if set(declared)!=set(DATA_EPOCH_ASSETS):issues.append("freeze asset set mismatch")

    epoch_payload={
        "schema_version":1,
        "family":NEW_EXACT_DEV_EXTENSION_FAMILY,
        "freeze_id":contract.get("freeze_id"),
        "role":"DEV_EXTENSION",
        "counts":counts,
        "minimum_support":DATA_EPOCH_MINIMUM,
        "data_contract":DATA_EPOCH_EXACT_CONTRACT,
        "files":assets,
        "selection_uses_labels":False,
    }
    epoch_sha=digest_bytes(canonical(epoch_payload))
    if contract and contract.get("data_epoch_sha256")!=epoch_sha:
        issues.append("freeze data_epoch_sha256 differs from canonical controller digest")
    binding_sha=digest_bytes(canonical({
        "v36_dev_sha256":EXPECTED_DEV_SHA,
        "data_epoch_sha256":epoch_sha,
        "freeze_contract_sha256":contract_sha,
    }))
    ready=not issues
    return {
        "required":True,"status":"READY" if ready else "WAITING_DATA_EPOCH_FREEZE",
        "ready":ready,"issues":issues,"counts":counts,"minimum_support":DATA_EPOCH_MINIMUM,
        "support_checks":support,"freeze_id":contract.get("freeze_id"),
        "data_epoch_sha256":epoch_sha,"freeze_contract_sha256":contract_sha,
        "binding_sha256":binding_sha,"files":assets,"canonical_payload":epoch_payload,
        "seal_labels_loaded":False,"final_reserve_labels_loaded":False,
    }


def identical_data_epoch(before:dict[str,Any],after:dict[str,Any])->bool:
    return (after.get("ready") is True and
            after.get("binding_sha256")==before.get("binding_sha256") and
            after.get("files")==before.get("files") and
            after.get("counts")==before.get("counts") and
            after.get("freeze_id")==before.get("freeze_id"))


def process_alive(pid:int)->bool:
    if pid<=0:return False
    try:
        import psutil
        return bool(psutil.pid_exists(pid))
    except Exception:
        try:os.kill(pid,0);return True
        except OSError:return False


@contextmanager
def exclusive_lock():
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    if LOCK_PATH.exists():
        old=read_json(LOCK_PATH,{}) or {};pid=int(old.get("pid",-1))
        if process_alive(pid):
            raise RuntimeError(f"controller already active: pid={pid}")
        stale=LOCK_PATH.with_name(f"{LOCK_PATH.name}.stale.{int(time.time())}")
        os.replace(LOCK_PATH,stale)
    descriptor=os.open(LOCK_PATH,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,canonical({"pid":os.getpid(),"started":now()}));os.fsync(descriptor)
    finally:os.close(descriptor)
    try:yield
    finally:
        if LOCK_PATH.exists():LOCK_PATH.unlink()


class Controller:
    def __init__(self,start_version:int,resume:bool):
        self.state=read_json(STATE_PATH,{}) if resume else {}
        self.state.setdefault("schema_version",1)
        self.state.setdefault("current_version",start_version)
        self.state.setdefault("phase","INITIALIZING")
        self.state.setdefault("completed_tasks",[])
        self.state.setdefault("pending_tasks",[])
        self.state.setdefault("latest_metrics",{})
        self.state.setdefault("candidate_state","NONE")
        self.state.setdefault("source_pool_counts",{"US":0,"KR":0,"total":0})
        self.state.setdefault("journal_head",None)
        self.state.setdefault("last_completed_version",start_version-1)
        self.state.setdefault("used_material_fingerprints",[])
        self.ensure_registry()
        self.reconcile_commits()
        self.save("controller_start")

    def ensure_registry(self)->None:
        registry=read_json(REGISTRY_PATH)
        if not isinstance(registry,dict):
            registry={"schema_version":1,"created_at":now(),"jobs":[dict(row) for row in DEFAULT_DYNAMIC_JOBS]}
            atomic_json(registry,REGISTRY_PATH)
        else:
            jobs=registry.setdefault("jobs",[]);known={str(row.get("job_id")) for row in jobs}
            for default in DEFAULT_DYNAMIC_JOBS:
                if default["job_id"] not in known:jobs.append(dict(default))
            atomic_json(registry,REGISTRY_PATH)

    def valid_commit(self,directory:Path)->dict[str,Any]|None:
        commit=read_json(directory/"COMMIT.json")
        manifest=read_json(directory/"ARTIFACT_MANIFEST.json")
        if not isinstance(commit,dict) or not isinstance(manifest,dict):return None
        if sha256(directory/"ARTIFACT_MANIFEST.json")!=commit.get("manifest_sha256"):return None
        for name,metadata in manifest.get("files",{}).items():
            path=directory/name
            if not path.is_file() or sha256(path)!=metadata.get("sha256"):return None
        return commit

    def reconcile_commits(self)->None:
        """Treat validated atomic output commits as recovery authority."""
        commits=[]
        for directory in sorted(ROOT.glob("output_V*")):
            match=directory.name.removeprefix("output_V")
            if not match.isdigit() or int(match)<37:continue
            commit=self.valid_commit(directory)
            if commit:commits.append((int(match),commit))
        if commits:
            self.state["last_completed_version"]=max(version for version,_ in commits)
            self.state["used_material_fingerprints"]=sorted({str(commit.get("material_fingerprint"))
                for _,commit in commits if commit.get("material_fingerprint")})
        registry=read_json(REGISTRY_PATH,{}) or {};changed=False
        for job in registry.get("jobs",[]):
            assigned=job.get("assigned_version")
            if assigned is not None and any(version==int(assigned) for version,_ in commits):
                job["status"]="COMMITTED";changed=True
            elif job.get("status") in {"CLAIMED","RUNNING","COMMITTING"}:
                job["attempt_count"]=int(job.get("attempt_count",0))+1
                job["status"]="READY" if job["attempt_count"]<3 else "QUARANTINED"
                job["recovered_at"]=now();changed=True
        if changed:atomic_json(registry,REGISTRY_PATH)
        # A data-only checkpoint (for example V70) does not consume a research
        # version.  If no material runner is registered for the immediate next
        # integer, resume at the earliest explicitly assigned READY version.
        # This preserves both "no fake version" and crash recovery semantics.
        base_next=int(self.state["last_completed_version"])+1
        ready_assigned=sorted({int(job["assigned_version"])
            for job in registry.get("jobs",[])
            if job.get("status") in {"READY","RETRY_WAIT"}
            and job.get("assigned_version") is not None
            and int(job["assigned_version"])>=base_next
            and (ROOT/str(job.get("runner","")).strip()).is_file()})
        next_version=base_next if base_next in ready_assigned or not ready_assigned else ready_assigned[0]
        self.state["current_version"]=next_version
        if next_version>base_next:
            self.state["unconsumed_data_checkpoint_versions"]=list(range(base_next,next_version))
            self.state["version_gap_reason"]=(
                "No material runner was registered for these data-only checkpoint versions; "
                "research version numbers were not consumed."
            )
        self.reconcile_champion(commits)

    @staticmethod
    def champion_score(report:dict[str,Any])->tuple:
        selected=report.get("selected",{});metrics=selected.get("metrics",{});robust=selected.get("robustness",{})
        try:gate=canonical_research_gate(selected)
        except (KeyError,TypeError,ValueError):gate={"passed":0}
        bootstrap=robust.get("bootstrap",{});markets=metrics.get("by_market",{})
        return (int(gate.get("passed",0) or 0),float(bootstrap.get("balanced_accuracy_lower95",-1) or -1),
                float(bootstrap.get("highconf_strategy_net_lower95",-1) or -1),
                min(float(markets.get("US",{}).get("balanced_accuracy",-1) or -1),
                    float(markets.get("KR",{}).get("balanced_accuracy",-1) or -1)),
                float(metrics.get("balanced_accuracy",-1) or -1),float(metrics.get("auc",-1) or -1),
                float(metrics.get("strategy_mean_signed_net",-1) or -1))

    def reconcile_champion(self,commits:list[tuple[int,dict[str,Any]]])->None:
        candidates=[]
        # V58 is the immutable bootstrap champion even before dynamic commits.
        # Always rediscover every valid atomic commit.  Calling this after one
        # new version with only that version in ``commits`` must not erase an
        # older, stronger (or tie-preferred) champion such as V69.
        discovered=set()
        for directory in sorted(ROOT.glob("output_V*")):
            suffix=directory.name.removeprefix("output_V")
            if suffix.isdigit() and int(suffix)>=58 and self.valid_commit(directory):
                discovered.add(int(suffix))
        versions={58}|discovered|{version for version,_ in commits if version>=58}
        for version in sorted(versions):
            report=read_json(ROOT/f"output_V{version}"/"DEV_ROBUSTNESS_REPORT.json")
            if isinstance(report,dict):candidates.append((self.champion_score(report),version,report))
        if not candidates:return
        score,version,report=max(candidates,key=lambda row:(row[0],-row[1]))
        payload={"version":version,"updated_at":now(),"score":list(score),
                 "hypothesis":report.get("hypothesis"),"report":f"output_V{version}/DEV_ROBUSTNESS_REPORT.json"}
        atomic_json(payload,CHAMPION_PATH);self.state["current_champion"]=payload

    def journal(self,event:str,detail:Any)->None:
        payload={"timestamp":now(),"event":event,"detail":detail,
                 "previous_hash":self.state.get("journal_head")}
        payload["entry_hash"]=digest_bytes(canonical(payload))
        JOURNAL_PATH.parent.mkdir(parents=True,exist_ok=True)
        with JOURNAL_PATH.open("a",encoding="utf-8",newline="\n") as handle:
            handle.write(json.dumps(payload,ensure_ascii=False,sort_keys=True,default=str)+"\n")
            handle.flush();os.fsync(handle.fileno())
        self.state["journal_head"]=payload["entry_hash"]

    def save(self,event:str,detail:Any=None)->None:
        self.state["updated_at"]=now();self.journal(event,detail or {})
        atomic_json(self.state,STATE_PATH)

    def phase(self,name:str,**detail:Any)->None:
        self.state["phase"]=name;self.state.update(detail);self.save("phase",{"name":name,**detail})

    def verify_workspace(self)->dict[str,Any]:
        path=ROOT/"data"/"dev_contract_v36_labeled.csv.gz"
        observed=sha256(path)
        if observed!=EXPECTED_DEV_SHA:
            raise RuntimeError(f"V36 exact DEV hash mismatch: {observed}")
        if (ROOT/"cert"/"V36").exists():
            raise RuntimeError("cert/V36 unexpectedly exists; fail-closed before research")
        v35={}
        for candidate in sorted((ROOT/"cert"/"V35").rglob("*")):
            if candidate.is_file():v35[str(candidate.relative_to(ROOT))]=sha256(candidate)
        manifest={"dev_path":str(path.relative_to(ROOT)),"dev_sha256":observed,
                  "cert_v36_absent":True,"cert_v35_manifest_sha256":digest_bytes(canonical(v35)),
                  "runtime":runtime_limits.status()}
        baseline=self.state.get("immutable_baseline")
        if baseline and baseline!=manifest:
            raise RuntimeError("immutable V36/V35 baseline changed since controller initialization")
        self.state["immutable_baseline"]=manifest
        return manifest

    def program_for(self,version:int)->tuple[int,str,str,str]|None:
        legacy=next((row for row in LEGACY_RESEARCH_PROGRAM if row[0]==version),None)
        if legacy:return legacy
        registry=read_json(REGISTRY_PATH,{}) or {};jobs=registry.get("jobs",[])
        ready=[row for row in jobs if row.get("status") in {"READY","RETRY_WAIT"} and
               (row.get("assigned_version") in (None,version)) and (ROOT/str(row.get("runner","")).strip()).is_file()]
        if not ready:return None
        job=sorted(ready,key=lambda row:(-int(row.get("priority",0)),str(row.get("job_id"))))[0]
        job["assigned_version"]=version;job["status"]="CLAIMED";job["claimed_at"]=now()
        atomic_json(registry,REGISTRY_PATH)
        return (version,str(job["hypothesis"]),str(job["runner"]),str(job["description"]))

    def release_claim_for_data_epoch_wait(self,version:int)->None:
        """A missing freeze does not consume an attempt or a material version."""
        registry=read_json(REGISTRY_PATH,{}) or {};changed=False
        for job in registry.get("jobs",[]):
            if job.get("assigned_version")==version and job.get("status")=="CLAIMED":
                job["status"]="READY";job.pop("claimed_at",None);changed=True
        if changed:atomic_json(registry,REGISTRY_PATH)

    def prior_report(self,version:int)->dict[str,Any]:
        if version<=37:
            return read_json(ROOT/"output_V36"/"DEV_ROBUSTNESS_REPORT.json",{}) or {}
        champion=read_json(CHAMPION_PATH,{}) or {};champion_version=champion.get("version")
        if champion_version:return read_json(ROOT/f"output_V{champion_version}"/"DEV_ROBUSTNESS_REPORT.json",{}) or {}
        return read_json(ROOT/f"output_V{version-1}"/"DEV_ROBUSTNESS_REPORT.json",{}) or {}

    def acquisition_projection(self)->dict[str,Any]:
        roles=read_json(ROOT/"data"/"V59_DATA_ROLE_ASSIGNMENT.json",{}) or {}
        massive=read_json(ROOT/"data"/"MASSIVE_COVERAGE_AUDIT.json",{}) or {}
        kis=read_json(ROOT/"data"/"KIS_HISTORICAL_COVERAGE_AUDIT.json",{}) or {}
        status=read_json(STATE_DIR/"ACQUISITION_STATE.json",{}) or {}
        source_status=read_json(SOURCE_ACQUISITION_STATUS_PATH,{}) or {}
        exact=source_status.get("exact_eligible",{}) or {}
        exact_eligible={
            "US":int(exact.get("US",0) or 0),
            "KR":int(exact.get("KR",0) or 0),
            "total":int(exact.get("total",0) or 0),
        }
        if exact_eligible["total"]!=exact_eligible["US"]+exact_eligible["KR"]:
            raise RuntimeError("SOURCE_ACQUISITION_STATUS exact_eligible total mismatch")
        payload={"roles":roles.get("counts",{}),
                 "exact_eligible":exact_eligible,
                 "exact_eligible_by_role":source_status.get("exact_eligible_by_role",{}),
                 "research_seal_minimum_met":bool(source_status.get("research_seal_minimum_met",False)),
                 "final_meta_minimum_met":bool(source_status.get("final_meta_minimum_met",False)),
                 "massive_exact":massive.get("exact_contract_eligible",0),
                 "kis_exact":kis.get("queue_fetch",{}).get("exact_contract_eligible",0),
                 "phase":status.get("phase"),
                 "source_status_sha256":sha256(SOURCE_ACQUISITION_STATUS_PATH)
                    if SOURCE_ACQUISITION_STATUS_PATH.exists() else None,
                 "role_manifest_sha256":sha256(ROOT/"data"/"V59_DATA_ROLE_ASSIGNMENT.parquet")
                    if (ROOT/"data"/"V59_DATA_ROLE_ASSIGNMENT.parquet").exists() else None}
        payload["data_epoch_sha256"]=digest_bytes(canonical(payload));return payload

    def bottleneck(self,previous:dict[str,Any])->dict[str,Any]:
        selected=previous.get("selected",previous)
        metrics=selected.get("metrics",{}) if isinstance(selected,dict) else {}
        robust=selected.get("robustness",{}) if isinstance(selected,dict) else {}
        return {"diagnosed_at":now(),"previous_status":previous.get("status"),
                "balanced_accuracy":metrics.get("balanced_accuracy"),"auc":metrics.get("auc"),
                "edge":metrics.get("edge_vs_naive",metrics.get("edge")),
                "high_confidence_net":metrics.get("strategy_mean_signed_net"),
                "bootstrap_ba_lower":robust.get("bootstrap",{}).get("balanced_accuracy_lower95",
                    robust.get("bootstrap",{}).get("balanced_accuracy_lower")),
                "primary_bottlenecks":["weak directional generalization","source/regime transfer",
                    "robust high-confidence net return","fresh exact untouched source shortage"]}

    def identifiers(self,version:int,hypothesis:str,description:str,runner:Path,
                    data_epoch:dict[str,Any]|None=None)->dict[str,Any]:
        data_sha=EXPECTED_DEV_SHA
        extension_ids={}
        if is_new_exact_dev_extension_family(hypothesis):
            if not isinstance(data_epoch,dict) or not data_epoch.get("ready"):
                raise RuntimeError("NEW_EXACT_DEV_EXTENSION_RETRAIN requires a ready frozen data epoch")
            data_sha=str(data_epoch["binding_sha256"])
            extension_ids={
                "v36_dev_sha256":EXPECTED_DEV_SHA,
                "data_epoch_sha256":str(data_epoch["data_epoch_sha256"]),
                "data_epoch_contract_sha256":str(data_epoch["freeze_contract_sha256"]),
                "data_epoch_freeze_id":str(data_epoch["freeze_id"]),
            }
        feature_spec={"hypothesis":hypothesis,"material_change":description}
        validation={"outer":"4 expanding chronological OOF folds","embargo_minutes":35,
                    "selection":"inner only","seal_outcomes_loaded":False}
        model_spec={"runner":runner.name,"runner_sha256":sha256(runner)}
        material_fp=digest_bytes(canonical({"data_sha":data_sha,"feature_spec":feature_spec,
                                            "model_spec":model_spec,"validation":validation}))
        experiment_id=digest_bytes(canonical({"version":version,"material_fingerprint":material_fp}))
        execution_id=digest_bytes(canonical({"experiment_id":experiment_id,
            "controller_sha256":sha256(Path(__file__)),"python":sys.version}))
        return {"data_sha":data_sha,"feature_sha":digest_bytes(canonical(feature_spec)),
                "material_fingerprint":material_fp,"experiment_id":experiment_id,
                "execution_cache_id":execution_id,**extension_ids}

    def run_version(self,version:int)->dict[str,Any]:
        program=self.program_for(version)
        if not program:
            self.phase("WAITING_MATERIAL_INTERVENTION",current_version=version,
                       pending_tasks=["acquisition/data epoch change or genuinely new registered material job"])
            return {"status":"WAITING_QUEUE"}
        _,hypothesis,runner_name,description=program;runner=ROOT/runner_name
        data_epoch=None
        if is_new_exact_dev_extension_family(hypothesis):
            data_epoch=inspect_data_epoch_contract()
            if not data_epoch["ready"]:
                self.release_claim_for_data_epoch_wait(version)
                audit={key:value for key,value in data_epoch.items() if key!="canonical_payload"}
                self.phase("WAITING_DATA_EPOCH_FREEZE",current_version=version,hypothesis=hypothesis,
                           data_epoch_audit=audit,
                           pending_tasks=["freeze role-assigned exact DEV_EXTENSION at required support",
                                          str(DATA_EPOCH_CONTRACT_PATH.relative_to(ROOT))])
                return {"status":"WAITING_DATA_EPOCH_FREEZE","data_epoch_audit":audit}
        if not runner.exists():
            self.phase("RESEARCH_BLOCKED_IMPLEMENTATION",current_version=version,hypothesis=hypothesis,
                       pending_tasks=[f"implement {runner_name}; do not consume V{version}"])
            return {"status":"NO_MATERIAL_RUNNER","runner":runner_name}
        ids=self.identifiers(version,hypothesis,description,runner,data_epoch)
        committed=ROOT/f"output_V{version}"
        if committed.exists():
            commit=read_json(committed/"COMMIT.json")
            if commit and commit.get("experiment_id")==ids["experiment_id"]:
                self.journal("reuse_committed_version",{"version":version,**ids})
                return read_json(committed/"DEV_ROBUSTNESS_REPORT.json",{})
            raise RuntimeError(f"unrecognized existing output directory: {committed}")
        if ids["material_fingerprint"] in self.state["used_material_fingerprints"]:
            raise RuntimeError("material fingerprint already consumed; fake version prevented")
        attempt=f"{ids['execution_cache_id'][:16]}-{int(time.time())}"
        stage=STAGING_DIR/f"V{version}"/attempt;stage.mkdir(parents=True,exist_ok=False)
        previous=self.prior_report(version);bottleneck=self.bottleneck(previous)
        opened_evidence=("V36 exact DEV plus role-frozen exact DEV_EXTENSION"
                         if data_epoch is not None else "V36 exact DEV only")
        plan={"version":f"V{version}","created_at":now(),"hypothesis":hypothesis,
              "material_change":description,"runner":runner.name,**ids,
              "validation":"nested chronological; 35-minute embargo; inner-only policy selection",
              "opened_evidence":opened_evidence,"seal_outcomes_loaded":False,
              "final_reserve_outcomes_loaded":False}
        atomic_json(plan,stage/"VERSION_PLAN.json");atomic_json(bottleneck,stage/"BOTTLENECK_ANALYSIS.json")
        if data_epoch is not None:
            atomic_json({"status":"PRE_RUN_FROZEN","before":data_epoch,"after":None,
                         "identical_before_after":None},stage/"DATA_EPOCH_BINDING.json")
        atomic_json({"version":f"V{version}","material":True,"hypothesis":hypothesis,
                     "description":description,"fingerprint":ids["material_fingerprint"]},
                    stage/"MATERIAL_CHANGE.json")
        self.phase("RESEARCHING",current_version=version,hypothesis=hypothesis,
                   experiment_id=ids["experiment_id"],input_sha=ids["data_sha"],
                   completed_tasks=[],pending_tasks=[runner.name,"report commit"])
        environment=os.environ.copy();environment["MARKET_BIO_VERSION_OUTPUT"]=str(stage)
        if data_epoch is not None:
            environment["MARKET_BIO_DATA_EPOCH_SHA256"]=str(data_epoch["data_epoch_sha256"])
            environment["MARKET_BIO_DATA_EPOCH_FREEZE_ID"]=str(data_epoch["freeze_id"])
            environment["MARKET_BIO_DATA_EPOCH_CONTRACT"]=str(DATA_EPOCH_CONTRACT_PATH)
        log_path=stage/"RUN.log"
        registry=read_json(REGISTRY_PATH,{}) or {}
        for job in registry.get("jobs",[]):
            if job.get("assigned_version")==version and job.get("status")=="CLAIMED":
                job["status"]="RUNNING";job["started_at"]=now();job["runner_pid_parent"]=os.getpid()
        atomic_json(registry,REGISTRY_PATH)
        with log_path.open("w",encoding="utf-8",newline="\n") as log:
            completed=subprocess.run([sys.executable,str(runner)],cwd=ROOT,env=environment,
                                     stdout=log,stderr=subprocess.STDOUT,check=False)
            log.flush();os.fsync(log.fileno())
        if data_epoch is not None:
            data_epoch_after=inspect_data_epoch_contract()
            identical=identical_data_epoch(data_epoch,data_epoch_after)
            atomic_json({"status":"VERIFIED_IDENTICAL" if identical else "DATA_EPOCH_MUTATED",
                         "before":data_epoch,"after":data_epoch_after,
                         "identical_before_after":identical},stage/"DATA_EPOCH_BINDING.json")
            if not identical:
                failed=RESEARCH_DIR/"quarantine"/f"V{version}-{attempt}-data-epoch-mutated"
                failed.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,failed)
                raise RuntimeError(f"data epoch changed during {runner.name}; quarantined={failed}")
        if completed.returncode:
            failed=RESEARCH_DIR/"quarantine"/f"V{version}-{attempt}"
            failed.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,failed)
            raise RuntimeError(f"{runner.name} exit={completed.returncode}; quarantined={failed}")
        result=read_json(stage/"DEV_ROBUSTNESS_REPORT.json")
        if not isinstance(result,dict):raise RuntimeError("runner produced no robustness report")
        selected=result.get("selected",{})
        try:canonical_gate=canonical_research_gate(selected)
        except (KeyError,TypeError,ValueError) as error:
            failed=RESEARCH_DIR/"quarantine"/f"V{version}-{attempt}-invalid-research-gate"
            failed.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,failed)
            raise RuntimeError(f"cannot recompute canonical research gate: {error!r}; quarantined={failed}")
        reported_gate=selected.get("research_gate")
        gate_match=reported_gate==canonical_gate
        atomic_json({"status":"MATCH" if gate_match else "MISMATCH",
                     "reported":reported_gate,"canonical":canonical_gate,
                     "material_gate":selected.get("material_gate")},
                    stage/"CANONICAL_RESEARCH_GATE_AUDIT.json")
        if not gate_match:
            failed=RESEARCH_DIR/"quarantine"/f"V{version}-{attempt}-research-gate-mismatch"
            failed.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,failed)
            raise RuntimeError(f"runner research gate differs from canonical contract; quarantined={failed}")
        status=result.get("status","RESEARCH_FAIL")
        acquisition=self.acquisition_projection()
        exact_eligible=acquisition["exact_eligible"]
        source_pool={"audited_at":now(),"label_blind":True,"frozen":False,
                     "eligible_exact":exact_eligible,
                     "minimum":{"US":250,"KR":100,"total":500},
                     "status":"WAITING_SOURCE","seal_authorized":False,
                     "evidence":"data/V59_DATA_ROLE_ASSIGNMENT.json","acquisition":acquisition}
        data_status={"contract":"EXACT_T2_30M_V36","rows":9462,"US":7976,"KR":1486,
                     "dev_sha256":EXPECTED_DEV_SHA,"immutable":True,"seal_labels_loaded":False}
        if data_epoch is not None:
            data_status.update({"contract":"EXACT_T2_30M_V36_PLUS_FROZEN_DEV_EXTENSION",
                "combined_data_sha256":ids["data_sha"],"dev_extension":{
                    "freeze_id":data_epoch["freeze_id"],"counts":data_epoch["counts"],
                    "data_epoch_sha256":data_epoch["data_epoch_sha256"],
                    "freeze_contract_sha256":data_epoch["freeze_contract_sha256"]},
                "research_seal_labels_loaded":False,"final_reserve_labels_loaded":False})
        atomic_json(data_status,stage/"DATA_STATUS.json");atomic_json(source_pool,stage/"SOURCE_POOL_STATUS.json")
        atomic_json({"from":f"V{version-1}","to":f"V{version}","hypothesis":hypothesis,
                     "material_fingerprint":ids["material_fingerprint"],"status":status},
                    stage/"VERSION_DELTA.json")
        gate=canonical_gate
        robust=bool(gate["robust_survivor"])
        next_version=version+1
        next_program=self.program_for(next_version)
        next_action={"robust_survivor":robust,"seal_action":"WAIT_FRESH_SOURCE" if robust else "NONE",
                     "next_version":next_version,"next_hypothesis":next_program[1] if next_program else None,
                     "continue":True,"reason":"Individual research/source failure is not a stop condition."}
        atomic_json(next_action,stage/"NEXT_ACTION.json")
        run_status={"version":version,"status":status,"phase":"ROBUST_SURVIVOR" if robust else "RESEARCH_FAIL",
                    "exit_code":completed.returncode,"completed_at":now(),"experiment_id":ids["experiment_id"],
                    "seal_state":"UNOPENED","final_status":"CONTINUE"}
        atomic_json(run_status,stage/"RUN_STATUS.json")
        manifest={}
        for item in sorted(stage.iterdir()):
            if item.is_file() and item.name not in {"ARTIFACT_MANIFEST.json","COMMIT.json"}:
                manifest[item.name]={"sha256":sha256(item),"bytes":item.stat().st_size}
        atomic_json({"version":version,"experiment_id":ids["experiment_id"],"files":manifest},
                    stage/"ARTIFACT_MANIFEST.json")
        missing=[name for name in REQUIRED_OUTPUTS if not (stage/name).is_file()]
        if missing:raise RuntimeError(f"staging missing required artifacts: {missing}")
        commit={"version":version,"committed_at":now(),**ids,
                "manifest_sha256":sha256(stage/"ARTIFACT_MANIFEST.json")}
        atomic_json(commit,stage/"COMMIT.json")
        committed.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,committed)
        self.state["used_material_fingerprints"].append(ids["material_fingerprint"])
        self.state["last_completed_version"]=version
        self.state["completed_tasks"]=[runner.name,"reports","atomic commit","ledger"]
        self.state["pending_tasks"]=["fresh exact source acquisition",f"V{next_version} material research"]
        self.state["candidate_state"]="ROBUST_SURVIVOR" if robust else "RESEARCH_FAIL"
        self.state["source_pool_counts"]=dict(exact_eligible)
        self.state["latest_metrics"]=result.get("selected",{}).get("metrics",{})
        self.append_ledger(version,hypothesis,description,ids,result,run_status)
        registry=read_json(REGISTRY_PATH,{}) or {}
        for job in registry.get("jobs",[]):
            if job.get("assigned_version")==version:
                job["status"]="COMMITTED";job["experiment_id"]=ids["experiment_id"];job["completed_at"]=now()
        atomic_json(registry,REGISTRY_PATH)
        self.reconcile_champion([(version,commit)])
        self.save("version_committed",{"version":version,"output":str(committed),"status":status})
        return result

    def append_ledger(self,version:int,hypothesis:str,description:str,ids:dict[str,str],
                      result:dict[str,Any],status:dict[str,Any])->None:
        selected=result.get("selected",{});metrics=selected.get("metrics",{});robust=selected.get("robustness",{})
        bootstrap=robust.get("bootstrap",{});gate=selected.get("research_gate",{})
        market=metrics.get("by_market",{});folds=metrics.get("by_fold",{})
        worst_fold=min((value.get("balanced_accuracy",1.) for value in folds.values()),default=None)
        row={"version":version,"timestamp":now(),"hypothesis":hypothesis,
             "material_change":description,"data_sha":ids["data_sha"],"feature_sha":ids["feature_sha"],
             "US_model":"source-specific late fusion","KR_model":"source-specific late fusion",
             "source_models":"fold-minimum semantic experts","BA":metrics.get("balanced_accuracy"),
             "AUC":metrics.get("auc"),"edge":metrics.get("edge_vs_naive",metrics.get("edge")),
             "HC_coverage":metrics.get("highconf_coverage"),"HC_accuracy":metrics.get("highconf_accuracy"),
             "HC_net":metrics.get("strategy_mean_signed_net"),
             "bootstrap_BA_lower":bootstrap.get("balanced_accuracy_lower95"),
             "bootstrap_HC_net_lower":bootstrap.get("highconf_strategy_net_lower95"),
             "US_BA":market.get("US",{}).get("balanced_accuracy"),
             "KR_BA":market.get("KR",{}).get("balanced_accuracy"),
             "US_SEC_BA":metrics.get("by_source_family",{}).get("US_SEC",{}).get("balanced_accuracy"),
             "US_NEWS_BA":metrics.get("by_source_family",{}).get("US_NEWS",{}).get("balanced_accuracy"),
             "KR_NEWS_BA":metrics.get("by_source_family",{}).get("KR_NEWS",{}).get("balanced_accuracy"),
             "KR_KIND_BA":metrics.get("by_source_family",{}).get("KR_KIND",{}).get("balanced_accuracy"),
             "worst_fold":worst_fold,"research_gate_passed_count":gate.get("passed"),"seal_state":"UNOPENED",
             "final_status":status["final_status"],"experiment_id":ids["experiment_id"]}
        RESEARCH_DIR.mkdir(parents=True,exist_ok=True)
        with LEDGER_JSONL.open("a",encoding="utf-8",newline="\n") as handle:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True,default=str)+"\n")
            handle.flush();os.fsync(handle.fileno())
        rows=[]
        for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():rows.append(json.loads(line))
        temporary=LEDGER_CSV.with_name(LEDGER_CSV.name+f".tmp.{os.getpid()}")
        with temporary.open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,LEDGER_CSV)

    def run(self,continue_until_pass:bool,max_completed:int|None)->int:
        completed_this_run=0
        while True:
            self.phase("VERIFYING_INTEGRITY")
            integrity=self.verify_workspace();self.save("integrity_pass",integrity)
            if verify_final_marker():
                self.phase("FINAL_PASS");return 0
            version=int(self.state["current_version"])
            result=self.run_version(version)
            if result.get("status") in {"WAITING_QUEUE","WAITING_DATA_EPOCH_FREEZE"}:return 20
            completed_this_run+=1;self.state["current_version"]=version+1
            self.phase("CHECKPOINTED",current_version=version+1)
            if max_completed is not None and completed_this_run>=max_completed:return 0
            if not continue_until_pass:return 0


def verify_final_marker()->bool:
    marker=read_json(FINAL_MARKER)
    if not isinstance(marker,dict) or marker.get("status")!="FINAL_PASS":return False
    certificate=Path(marker.get("certificate",""))
    if not certificate.is_absolute():certificate=ROOT/certificate
    report=read_json(certificate)
    if not isinstance(report,dict):return False
    checks=report.get("checks",[])
    return len(checks)==12 and all(item.get("pass") is True for item in checks) and \
        report.get("contract_integrity") is True and report.get("no_leakage") is True and \
        sha256(certificate)==marker.get("certificate_sha256")


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--start-version",type=int,default=37)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--continue-until-pass",action="store_true")
    parser.add_argument("--max-completed-versions",type=int)
    return parser.parse_args()


def main()->int:
    args=parse_args();runtime_limits.configure()
    try:
        with exclusive_lock():
            return Controller(args.start_version,args.resume).run(
                args.continue_until_pass,args.max_completed_versions)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        STATE_DIR.mkdir(parents=True,exist_ok=True)
        failure={"timestamp":now(),"error":repr(error),"traceback":traceback.format_exc()}
        atomic_json(failure,STATE_DIR/"LAST_CONTROLLER_EXCEPTION.json")
        print(json.dumps(failure,ensure_ascii=False,indent=2),file=sys.stderr);return 10


if __name__=="__main__":raise SystemExit(main())
