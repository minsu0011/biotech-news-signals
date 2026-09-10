"""One-shot historical mini-seal: frozen V224 only, fixed 60 IDs, zero tuning."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
import v224_prospective_locked_evaluation as locked
import v224_prospective_causal_one_shot as causal
from v224_historical_seal_guard import SealAccessGuard, digest, utc_now

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cert_research/V224_HISTORICAL_MINI_SEAL"
OUTCOME_COLS = ["y", "entry_price", "actual_entry_time_utc", "exit_price",
                "actual_exit_time_utc", "fwd_ret_30m", "actual_hold_seconds"]


def save_json(name, value):
    return locked.write_immutable(OUT/name, locked.pretty_json_bytes(causal.clean_json(value)))


def members():
    m = pd.read_csv(OUT/"PRECOMMITTED_MEMBERSHIP.csv", dtype=str)
    expected = (OUT/"PRECOMMITTED_MEMBERSHIP_SHA256.txt").read_text().strip()
    if digest(OUT/"PRECOMMITTED_MEMBERSHIP.csv") != expected or len(m) != 60:
        raise RuntimeError("MEMBERSHIP_INTEGRITY_FAIL")
    return m


def strict_feature(event, price_path, exact_entry):
    """Preserve existing formulas while projecting just the metadata-selected entry open."""
    original = causal.projected_parquet
    projections=[]
    def project(path, columns, filters):
        if columns == ["timestamp", "open"]:
            filters = [("timestamp", "==", exact_entry)]
        elif columns == ["timestamp", "open", "high", "low", "close", "volume"]:
            if ("timestamp", "<", exact_entry) not in filters:
                raise RuntimeError("PREHISTORY_FILTER_NOT_STRICTLY_PRIOR")
        else:
            raise RuntimeError("UNAUTHORIZED_FEATURE_PROJECTION")
        frame = original(path, columns, filters)
        if not frame.empty and frame.timestamp.max() > exact_entry:
            raise RuntimeError("FEATURE_FUTURE_TIMESTAMP")
        projections.append({"columns":columns,"rows":len(frame),"filters":str(filters)})
        return frame
    causal.projected_parquet=project
    try:
        feature, reason, audit=causal.causal_feature_snapshot(event,price_path)
    finally:
        causal.projected_parquet=original
    if feature is not None:
        feature["price_source"] = str(event.price_provider) + "_1M_CAUSAL_PROJECTION"
    audit["strict_projections"] = projections
    audit["entry_open_rows_loaded"] = sum(p["rows"] for p in projections if p["columns"] == ["timestamp","open"])
    return feature,reason,audit


def prepare_features():
    membership=members()
    metadata=pd.read_csv(OUT/"CANDIDATE_MEMBERSHIP_METADATA.csv",dtype=str).set_index("canonical_event_id")
    timing=json.loads((OUT/"EXACT_METADATA_CONTRACT_AUDIT.json").read_text())
    times={r["canonical_event_id"]:r for r in timing["rows"]}
    text_path=OUT/"EXACT_ACCESSION_TEXT_AUDIT_RECOVERY.json"
    if not text_path.exists():text_path=OUT/"EXACT_ACCESSION_TEXT_AUDIT.json"
    texts=json.loads(text_path.read_text(encoding="utf-8"))
    if not texts["passed"]:
        raise RuntimeError("EXACT_ACCESSION_TEXT_INCOMPLETE_NO_OUTCOMES_ALLOWED")
    text_by_id={r["event_id"]:r for r in texts["rows"]}
    output=[]
    for row in membership.itertuples(index=False):
        identity=row.canonical_event_id
        item=text_by_id[identity]
        path=ROOT/item["text_path"]
        if digest(path)!=item["text_sha256"]:
            raise RuntimeError("TEXT_HASH_MISMATCH")
        event=pd.Series({"event_id":identity,"event_time_utc":row.event_time,"ticker":row.ticker,
                         "headline":item["headline"],"body":path.read_text(encoding="utf-8"),
                         "form":item["form"],"event_type":item["event_type"],"source":item["source"],
                         "price_provider":metadata.at[identity,"price_provider"]})
        price_path=ROOT/metadata.at[identity,"price_path"]
        feature,reason,audit=strict_feature(event,price_path,pd.Timestamp(times[identity]["actual_entry_time_utc"]))
        if feature is None:
            save_json("FEATURE_CONTRACT_FAILURE.json", {"event_id":identity,"reason":reason,"outcomes_opened":0})
            raise RuntimeError("CERT_ROW_CONTRACT_INVALID:"+reason)
        output.append({"event_id":identity,"feature_row":feature,"causal_audit":audit,
                       "price_path":str(price_path.relative_to(ROOT)),"price_sha256":digest(price_path),
                       "actual_exit_metadata":times[identity]["actual_exit_time_utc"],
                       "text_sha256":item["text_sha256"]})
    save_json("CAUSAL_FEATURE_SNAPSHOTS.json", {"rows":output,"outcome_rows_read":0,
              "formula_source_sha256":digest(ROOT/"v224_prospective_causal_one_shot.py"),
              "all_entry_projections_single_open":all(r["causal_audit"]["entry_open_rows_loaded"]==1 for r in output)})
    print(json.dumps({"features_ready":len(output),"outcome_rows_read":0}),flush=True)


def outcome_for(record):
    feature=record["feature_row"]
    path=ROOT/record["price_path"]
    if digest(path)!=record["price_sha256"]:
        raise RuntimeError("PRICE_SOURCE_MUTATED")
    exit_time=pd.Timestamp(record["actual_exit_metadata"])
    frame=pd.read_parquet(path,columns=["timestamp","open"],filters=[("timestamp","==",exit_time)])
    if len(frame)!=1:
        raise RuntimeError("CERT_ROW_CONTRACT_INVALID:EXIT_ROW_CARDINALITY")
    actual=pd.Timestamp(frame.timestamp.iloc[0])
    entry_time=pd.Timestamp(feature["actual_entry_time_utc"])
    hold=(actual-entry_time).total_seconds()
    entry=float(feature["entry_price"]); exit_price=float(frame.open.iloc[0])
    if not (1800<=hold<=1860 and np.isfinite(exit_price) and exit_price>0 and np.isfinite(entry) and entry>0):
        raise RuntimeError("CERT_ROW_CONTRACT_INVALID:PRICE_OR_HOLD")
    return {"event_id":record["event_id"],"y":int(exit_price>entry),"entry_price":entry,
            "actual_entry_time_utc":entry_time.isoformat(),"exit_price":exit_price,
            "actual_exit_time_utc":actual.isoformat(),"fwd_ret_30m":exit_price/entry-1,
            "actual_hold_seconds":hold}


def validate_preopen(membership, by_id):
    if digest(OUT/"CERT_WORKING_STATE_INITIAL.json")!=digest(ROOT/locked.FROZEN_REL/locked.FROZEN_STATE_NAME):
        raise RuntimeError("INITIAL_STATE_COPY_MISMATCH")
    clearance=json.loads((OUT/"CONTAMINATION_CLEARANCE.json").read_text())
    if clearance["status"]!="PASS" or clearance["identity_or_accession_overlap"]!=0:
        raise RuntimeError("FAIL_CLOSED_SEAL_CONTAMINATION")
    boundaries=[]
    for block in (1,2):
        previous=membership.loc[membership.block_id.eq(str(block)),"canonical_event_id"]
        following=membership.loc[membership.block_id.eq(str(block+1)),"canonical_event_id"]
        last_exit=max(pd.Timestamp(by_id[i]["actual_exit_metadata"]) for i in previous)
        first_decision=min(pd.Timestamp(by_id[i]["feature_row"]["actual_entry_time_utc"]) for i in following)
        if last_exit>first_decision:raise RuntimeError("BLOCK_STATE_USES_UNAVAILABLE_OUTCOME")
        boundaries.append({"completed_block":block,"last_exit":last_exit.isoformat(),
                           "next_first_decision":first_decision.isoformat(),"causal":True})
    for record in by_id.values():
        if digest(ROOT/record["price_path"])!=record["price_sha256"]:
            raise RuntimeError("PRICE_SOURCE_MUTATED_BEFORE_OPEN")
    save_json("PRE_OPEN_READINESS.json",{"passed":True,"outcomes_opened":0,"block_boundaries":boundaries,
              "membership_sha":digest(OUT/"PRECOMMITTED_MEMBERSHIP.csv"),
              "feature_snapshot_sha":digest(OUT/"CAUSAL_FEATURE_SNAPSHOTS.json"),
              "clearance_sha":digest(OUT/"CONTAMINATION_CLEARANCE.json")})


def final_report(scored,blocks,guard,runtime,reason=None):
    m=members(); counts=guard.counters()
    pre=json.loads((OUT/"PRE_CERT_INTEGRITY.json").read_text())
    post={name:digest(ROOT/locked.FROZEN_REL/name) for name in pre["files"]}
    unchanged=post==pre["files"]
    frozen_code=json.loads((OUT/"PRE_OPEN_EXECUTION_CODE_FREEZE.json").read_text())
    code_unchanged=all(digest(ROOT/p)==sha for p,sha in frozen_code.items())
    counters_valid=(counts["unique_outcome_rows_opened"]==60 and counts["completed_blocks"]==3
                    and counts["FINAL_META_OUTCOME_ROWS_READ"]==0
                    and counts["NON_US_SEC_RESEARCH_SEAL_OUTCOME_ROWS_READ"]==0
                    and counts["REMAINING_US_SEC_OUTCOME_ROWS_READ"]==0)
    role_sha=digest(ROOT/"data/V59_DATA_ROLE_ASSIGNMENT.parquet")
    inventory=json.loads((OUT/"METADATA_PREFLIGHT.json").read_text())
    reserves=[]
    for row in inventory["inventory_metadata_only"]:
        if row["role"] not in {"RESEARCH_SEAL_POOL","FINAL_META_RESERVE"}:continue
        remaining=row["rows"]
        if row["role"]=="RESEARCH_SEAL_POOL" and row["source_family"]=="US_SEC":remaining-=len(guard.opened)
        reserves.append({**row,"outcome_rows_still_unopened":remaining})
    save_json("UNOPENED_RESERVE_MANIFEST.json",{"metadata_counts":reserves,"original_role_registry_unchanged":role_sha==inventory["role_assignment_sha256"],
              "opened_membership_ids_count":len(guard.opened),"opening_authority":"SEAL_OPEN_LEDGER.csv and OUTCOME_ACCESS_JOURNAL.jsonl",
              "all_final_meta_unopened":True,"all_non_us_sec_research_seal_unopened":True})
    metrics=locked.metrics_for(scored,runtime.scorer.COST,include_bootstrap=True) if len(scored) else {}
    worst=min((b["metrics"]["balanced_accuracy"] for b in blocks if b["metrics"]["balanced_accuracy"] is not None),default=None)
    bootstrap=metrics.get("bootstrap",{})
    def ge(x,t):return x is not None and np.isfinite(x) and x>=t
    direction={"BA_ge_054":ge(metrics.get("balanced_accuracy"),.54),"AUC_ge_056":ge(metrics.get("auc"),.56),
               "edge_ge_002":ge(metrics.get("edge_vs_naive"),.02),"bootstrap_BA_L95_ge_050":ge(bootstrap.get("balanced_accuracy_lower95"),.5),
               "worst_block_BA_ge_049":ge(worst,.49)}
    hc={"coverage_ge_015":ge(metrics.get("highconf_coverage"),.15),"accuracy_ge_060":ge(metrics.get("highconf_accuracy"),.6),
        "net_gt_0":metrics.get("highconf_net") is not None and metrics["highconf_net"]>0}
    economic=bootstrap.get("highconf_net_lower95") is not None and bootstrap["highconf_net_lower95"]>0
    integrity={"PRE_CERT_SHA":pre["files"],"POST_CERT_SHA":post,"frozen_artifacts_unchanged":unchanged,
               "execution_code_unchanged":code_unchanged,"access_counter_checks_pass":counters_valid,
               "role_registry_unchanged":role_sha==inventory["role_assignment_sha256"],"counters":counts,
               "retuning_count":0,"model_retraining_count":0,"model_selection_on_seal":False,
               "certification_feature_and_outcome_access_no_leakage":reason is None,
               "historical_asof_training_independence":False,
               "temporal_limitation":"Frozen end-of-DEV training/state is later than historical holdout events; not an as-of deployment backtest.",
               "all_integrity_checks_pass":unchanged and code_unchanged and counters_valid and role_sha==inventory["role_assignment_sha256"] and reason is None and len(scored)==60}
    save_json("INTEGRITY_AUDIT.json",integrity)
    metrics.update({"worst_block_BA":worst,"direction_checks":direction,"hc_point_checks":hc,"economic_robust_pass":economic,
                    "full_60_evaluated":len(scored)==60,"blocks":blocks})
    save_json("OVERALL_METRICS.json",metrics)
    dev=json.loads((ROOT/"output_V224/V224_CAUSAL_POLARITY_CONFIDENCE_GUARD_REPORT.json").read_text())["results"]["V224_EXACT_SEC_N_T_S_GUARDED"]["overall"]
    delta={k:{"DEV":dev[k],"mini_seal":metrics.get(k),"delta":metrics[k]-dev[k] if metrics.get(k) is not None else None}
           for k in ["balanced_accuracy","auc","highconf_accuracy"]}
    save_json("GENERALIZATION_DELTA.json",delta)
    raw_metrics={}
    if len(scored):
        raw=scored.copy();raw["probability"]=raw.raw_probability;raw["prediction"]=raw.raw_probability.ge(.5)
        # Direction-only diagnostic: no challenger selection, no redefinition of HC.
        from sklearn.metrics import balanced_accuracy_score,roc_auc_score
        raw_metrics={"accuracy":float((raw.prediction==raw.y.astype(bool)).mean()),
                     "balanced_accuracy":float(balanced_accuracy_score(raw.y,raw.prediction)),
                     "auc":float(roc_auc_score(raw.y,raw.probability)) if raw.y.nunique()==2 else None}
        save_json("RAW_POLARITY_DIAGNOSTIC.json",{"raw":raw_metrics,"primary":{"balanced_accuracy":metrics["balanced_accuracy"],"auc":metrics["auc"]},
                  "primary_BA_minus_raw":metrics["balanced_accuracy"]-raw_metrics["balanced_accuracy"],"used_for_selection":False})
    passed=all(direction.values()) and all(hc.values()) and integrity["all_integrity_checks_pass"]
    status="V224_HISTORICAL_MINI_SEAL_PASS" if passed else "V224_HISTORICAL_MINI_SEAL_FAIL"
    fmt=lambda x:"NOT_EVALUATED" if x is None else f"{x:.6f}" if isinstance(x,float) else str(x)
    lines=["# OPENED SEAL REGION", "", "Role: RESEARCH_SEAL (workspace: RESEARCH_SEAL_POOL)", "Source: US_SEC",
           f"Selection: chronological first 60 full metadata exact-eligible rows; blocks 20 / 20 / 20",
           f"Date range: {m.event_time.iloc[0]} through {m.event_time.iloc[-1]}",
           f"Unique tickers: {m.ticker.nunique()}; unique issuers (CIK): {m.CIK.nunique()}",
           f"Membership SHA: {guard.membership_sha}", f"Outcome rows opened: {len(guard.opened)}; completed evaluation rows: {len(scored)}",
           "Outcome columns: "+", ".join(OUTCOME_COLS), "", "# UNOPENED RESERVE", ""]
    lines += [f"- {r['role']} / {r['source_family']}: {r['outcome_rows_still_unopened']} unopened" for r in reserves]
    lines += ["", "# RESULT", "",status,f"Fail-closed reason: {reason or 'NONE'}","",
              f"BA: {fmt(metrics.get('balanced_accuracy'))}; AUC: {fmt(metrics.get('auc'))}; edge: {fmt(metrics.get('edge_vs_naive'))}",
              f"Accuracy: {fmt(metrics.get('accuracy'))}; HC coverage: {fmt(metrics.get('highconf_coverage'))}; HC accuracy: {fmt(metrics.get('highconf_accuracy'))}; HC net: {fmt(metrics.get('highconf_net'))}",
              f"Bootstrap BA L95: {fmt(bootstrap.get('balanced_accuracy_lower95'))}; bootstrap HC net L95: {fmt(bootstrap.get('highconf_net_lower95'))}; worst block BA: {fmt(worst)}",
              f"DIRECTION PASS: {all(direction.values())}; HC POINT PASS: {all(hc.values())}; ECONOMIC ROBUST PASS: {economic}",
              "", "# BLOCK DRIFT", "", "| Block | First date | Last date | Prior raw AUC | Reversed | Confidence | BA | AUC | HC n / accuracy / net |", "|---|---|---|---|---|---|---|---|---|"]
    for b in blocks:lines.append(f"| {b['block']} | {b['first_date']} | {b['last_date']} | {fmt(b['policy']['prior_raw_auc'])} | {b['policy']['polarity_reversed']} | {b['policy']['confidence_polarity']} | {fmt(b['metrics']['balanced_accuracy'])} | {fmt(b['metrics']['auc'])} | {b['metrics']['highconf_n']} / {fmt(b['metrics']['highconf_accuracy'])} / {fmt(b['metrics']['highconf_net'])} |")
    raw_delta=metrics.get("balanced_accuracy",0)-raw_metrics.get("balanced_accuracy",0) if raw_metrics else None
    questions=[
        ("DEV performance reproduced on the historical holdout?",f"Core gates {'PASS' if passed else 'FAIL'}; metric deltas below. Not proof of equivalent performance or as-of-date independence."),
        ("BA >= .54?",str(direction["BA_ge_054"])),
        ("AUC >= .56?",str(direction["AUC_ge_056"])),
        ("Edge >= +2 percentage points?",str(direction["edge_ge_002"])),
        ("All 20-row blocks avoid the specified failure floor?",f"{direction['worst_block_BA_ge_049']}; worst BA={fmt(worst)}; the predefined floor is .49."),
        ("Did frozen polarity adaptation help versus raw?",f"Primary minus raw BA={fmt(raw_delta)}; raw BA={fmt(raw_metrics.get('balanced_accuracy'))}, AUC={fmt(raw_metrics.get('auc'))}. Descriptive only, not a model-selection test."),
        ("Did confidence state retain useful discrimination?",f"HC point gates={all(hc.values())}; confidence/correctness AUC={fmt(metrics.get('confidence_correctness_auc'))}. States: {', '.join(sorted({b['policy']['confidence_polarity'] for b in blocks}))}. N=60 does not establish stability across states."),
        ("HC accuracy >= 60%?",str(hc["accuracy_ge_060"])),
        ("HC mean net positive?",str(hc["net_gt_0"])),
        ("Economic bootstrap robustness confirmed?",f"{economic}; HC net L95={fmt(bootstrap.get('highconf_net_lower95'))}.")]
    lines += ["", "# TEN REQUIRED QUESTIONS", ""]
    lines += [f"{i}. {q} {a}" for i,(q,a) in enumerate(questions,1)]
    lines += ["", "# GENERALIZATION", "",json.dumps(delta,indent=2),"",
              "N=60 is limited historical held-out evidence, not full official certification or prospective confirmation.",
              integrity["temporal_limitation"],"", "NO RETUNING WAS PERFORMED. NO ADDITIONAL SEAL ROWS WERE OPENED.",
              "Opened IDs are now OPENED_HISTORICAL_CERT_EVIDENCE. Consult this ledger before any future seal use."]
    report_sha=locked.write_immutable(OUT/"FINAL_REPORT.md",("\n".join(lines)+"\n").encode())
    save_json("V224_HISTORICAL_MINI_SEAL_CERTIFICATE.json",{
        "model_id":"V224_EXACT_SEC_N_T_S_GUARDED","model_sha":pre["files"][locked.FROZEN_MODEL_NAME],
        "certification_type":"HISTORICAL_MINI_SEAL","seal_role":"RESEARCH_SEAL","opened_source":"US_SEC",
        "opened_rows":len(guard.opened),"blocks":len(blocks),"direction_pass":all(direction.values()),"hc_point_pass":all(hc.values()),
        "economic_robust_pass":economic,"overall_status":status,"failure_reason":reason,
        "remaining_seal_untouched":True,"final_meta_untouched":True,"membership_sha":guard.membership_sha,"report_sha":report_sha,
        "historical_asof_training_independence":False,"retuning_count":0})
    remaining_sec=sum(r["outcome_rows_still_unopened"] for r in reserves if r["role"]=="RESEARCH_SEAL_POOL" and r["source_family"]=="US_SEC")
    console=["="*52,"V224 HISTORICAL MINI-SEAL CERTIFICATION","="*52,"",
             "OPENED:","Research Seal / US_SEC",f"Rows: {len(guard.opened)}",f"Blocks completed: {len(blocks)} x 20",
             f"Date range: {m.event_time.iloc[0]} through {m.event_time.iloc[-1]}",f"Membership SHA: {guard.membership_sha}","",
             "UNOPENED:",f"Research Seal / US_SEC remaining: {remaining_sec}",
             "Research Seal / non-US_SEC: ALL UNOPENED","Final Meta: ALL UNOPENED","","RESULT:"]
    console += [f"{label}: {fmt(value)}" for label,value in [
        ("BA",metrics.get("balanced_accuracy")),("AUC",metrics.get("auc")),("EDGE",metrics.get("edge_vs_naive")),
        ("HC coverage",metrics.get("highconf_coverage")),("HC accuracy",metrics.get("highconf_accuracy")),("HC net",metrics.get("highconf_net")),
        ("Bootstrap BA L95",bootstrap.get("balanced_accuracy_lower95")),("Bootstrap HC net L95",bootstrap.get("highconf_net_lower95")),
        ("Worst block BA",worst),("DIRECTION PASS",all(direction.values())),("HC POINT PASS",all(hc.values())),("ECONOMIC ROBUST PASS",economic)]]
    console += ["","FINAL STATUS:",status,"","NO RETUNING WAS PERFORMED.","NO ADDITIONAL SEAL ROWS WERE OPENED.","="*52]
    locked.write_immutable(OUT/"CONSOLE_SUMMARY.txt",("\n".join(console)+"\n").encode())
    print("\n".join(console),flush=True)


def evaluate():
    membership=members()
    guard=SealAccessGuard(OUT)
    runtime=locked.load_frozen_runtime(ROOT)
    state=copy.deepcopy(runtime.initial_state)
    snapshots=json.loads((OUT/"CAUSAL_FEATURE_SNAPSHOTS.json").read_text(encoding="utf-8"))
    by_id={r["event_id"]:r for r in snapshots["rows"]}
    if len(by_id)!=60 or set(by_id)!=set(membership.canonical_event_id):raise RuntimeError("FEATURE_MEMBERSHIP_INVALID")
    validate_preopen(membership,by_id)
    code_paths=[Path(__file__)] + [ROOT/name for name in [
        "v224_historical_seal_guard.py","v224_prospective_causal_one_shot.py","v224_prospective_locked_evaluation.py",
        "v224_historical_mini_seal_preaudit.py","v224_historical_seal_contamination_audit.py",
        "v224_historical_mini_seal_prepare.py","test_v224_historical_seal_guard.py",
        "test_v224_historical_mini_seal_run.py"]]
    save_json("PRE_OPEN_EXECUTION_CODE_FREEZE.json",{str(p.relative_to(ROOT)):digest(p) for p in code_paths})
    all_scored=[];blocks=[];reason=None
    try:
        for block in (1,2,3):
            locked.verify_frozen_files(ROOT)
            ids=membership.loc[membership.block_id.eq(str(block)),"canonical_event_id"].tolist()
            features=pd.DataFrame([by_id[i]["feature_row"] for i in ids])
            projected=["event_id","event_time_utc","source_family","headline","body","form","event_type","regular_session_eligible"]
            projected += [c for c in runtime.scorer.NUMERIC_COLS if c in features and c not in projected]
            features=features[projected]
            features_sha=locked.write_immutable(OUT/f"BLOCK_{block:02d}_FEATURES_FROZEN.csv",features.to_csv(index=False,lineterminator="\n").encode())
            before=runtime.scorer.state_fingerprint(state)
            scored=runtime.scorer.score_block(features,runtime.bundle,state)
            second=runtime.scorer.score_block(features.sample(frac=1,random_state=224),runtime.bundle,state)
            try:
                pd.testing.assert_frame_equal(scored,second,check_exact=True)
            except AssertionError as error:
                raise RuntimeError("FAIL_CLOSED_NONDETERMINISTIC") from error
            if runtime.scorer.state_fingerprint(state)!=before:raise RuntimeError("SCORER_MUTATED_STATE")
            policy=runtime.scorer.derive_policy(state)
            public=scored.copy();public.insert(0,"canonical_event_id",public.event_id)
            public["raw_direction"]=public.raw_probability.ge(.5);public["scorer_sha"]=digest(ROOT/locked.FROZEN_REL/locked.FROZEN_SCORER_NAME)
            public["feature_sha"]=features_sha;public["prediction_timestamp"]=utc_now()
            prediction_sha=locked.write_immutable(OUT/f"BLOCK_{block:02d}_PREDICTIONS_FROZEN.csv",public.to_csv(index=False,lineterminator="\n").encode())
            locked.write_immutable(OUT/f"BLOCK_{block:02d}_PREDICTIONS_SHA256.txt",(prediction_sha+"\n").encode())
            save_json(f"BLOCK_{block:02d}_PREDICTION_FREEZE.json",{"prediction_sha":prediction_sha,"membership_sha":guard.membership_sha,
                      "deterministic":True,"state_fingerprint":before,"feature_sha":features_sha,"policy":policy,"created_at":utc_now()})
            guard.authorize(block)
            opened=[guard.open_one(i,lambda event_id:outcome_for(by_id[event_id])) for i in ids]
            outcomes=pd.DataFrame(opened)
            locked.write_immutable(OUT/f"BLOCK_{block:02d}_OPENED_OUTCOMES.csv",outcomes.to_csv(index=False,lineterminator="\n").encode())
            guard.finish_block(OUTCOME_COLS)
            merged=scored.merge(outcomes,on="event_id",validate="one_to_one")
            metric=locked.metrics_for(merged,runtime.scorer.COST)
            save_json(f"BLOCK_{block:02d}_METRICS.json",metric)
            blocks.append({"block":block,"first_date":str(merged.event_time_utc.min()),"last_date":str(merged.event_time_utc.max()),"policy":policy,"metrics":metric})
            all_scored.append(merged)
            state=runtime.scorer.complete_block(state,scored,outcomes)
            save_json(f"CERT_WORKING_STATE_AFTER_BLOCK_{block:02d}.json",state)
            print(json.dumps({"block_completed":block,"outcomes_opened":len(guard.opened)}),flush=True)
    except Exception as error:
        reason=f"{type(error).__name__}:{error}"
        save_json("FAIL_CLOSED_ERROR.json",{"reason":reason,"counters":guard.counters(),"no_retry":True})
    final_report(pd.concat(all_scored,ignore_index=True) if all_scored else pd.DataFrame(),blocks,guard,runtime,reason)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-features",action="store_true");parser.add_argument("--evaluate",action="store_true")
    args=parser.parse_args()
    if args.prepare_features==args.evaluate:parser.error("select exactly one phase")
    if args.prepare_features:prepare_features()
    else:evaluate()
