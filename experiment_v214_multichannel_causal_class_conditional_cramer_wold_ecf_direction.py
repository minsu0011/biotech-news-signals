"""V214 class-conditional Cramer-Wold empirical characteristic direction."""
from __future__ import annotations
import os
CONTROLLER_OUTPUT=os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"): os.environ[_n]="2"
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import experiment_v198_multichannel_causal_score_matching_energy_direction as scaffold

core=scaffold.scaffold; base=scaffold.base; ROOT=scaffold.ROOT; VERSION=214
HYPOTHESIS="MULTICHANNEL_CAUSAL_CLASS_CONDITIONAL_CRAMER_WOLD_ECF_DIRECTION_V1"
DEFAULT_OUT=ROOT/"research"/"staging"/"V214"/"LOCAL_FORBIDDEN"; FEATURES=scaffold.FEATURES
SCAFFOLD_PATH=ROOT/"experiment_v198_multichannel_causal_score_matching_energy_direction.py"; EXPECTED_SCAFFOLD_SHA256="80338c3f473670813ee1a888958b44911d9e926ba469a455ed48b086514ef9bc"
CHANNEL_N=8; FREQUENCIES=np.array([.5,1.,2.]); PROJECTION_N=64; FEATURE_N=384; TEMP_FLOOR=.05; SEED=21401
ARCHITECTURES=({"name":"CRAMER_WOLD_ECF_W0.25","weight":.25},{"name":"CRAMER_WOLD_ECF_W0.50","weight":.50})
require=scaffold.require; sha256=scaffold.sha256; array_sha256=scaffold.array_sha256; clean=scaffold.clean; bool_series=scaffold.bool_series; atomic_json=scaffold.atomic_json; atomic_csv=scaffold.atomic_csv
load_authorized=scaffold.load_authorized; metric=scaffold.metric; controller=scaffold.controller; numeric=scaffold.numeric; observed_past_to_recent_path=scaffold.observed_past_to_recent_path; equal_event_group_path=scaffold.equal_event_group_path; TrainChannelScale=scaffold.TrainChannelScale; blend_probability=scaffold.blend_probability

def fixed_directions():
    out=[]
    for i in range(CHANNEL_N):
        v=np.zeros(CHANNEL_N); v[i]=1.; out.append(v)
    for i in range(CHANNEL_N):
        for j in range(i+1,CHANNEL_N):
            v=np.zeros(CHANNEL_N); v[i]=v[j]=1/np.sqrt(2); out.append(v)
    for i in range(CHANNEL_N):
        for j in range(i+1,CHANNEL_N):
            v=np.zeros(CHANNEL_N); v[i]=1/np.sqrt(2); v[j]=-1/np.sqrt(2); out.append(v)
    result=np.stack(out); require(result.shape==(PROJECTION_N,CHANNEL_N),"V214 directions"); return result
DIRECTIONS=fixed_directions()
STATIC_SPEC={"feature_dimension":FEATURE_N,"state":"eight train-scaled channel horizon medians","projection_bank":"8 axes + 28 normalized pair sums + 28 normalized pair differences","frequencies":FREQUENCIES.tolist(),"representation":"class-conditional empirical characteristic-function means","score":"direct squared ECF-prototype distance contrast with train-IQR temperature","head":"none","target_model_fit":False}

def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V198 scaffold changed"); a=scaffold.verify_authority(); a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V198 output"}; a["v214_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False}; return a

def ecf_features(state):
    projection=state@DIRECTIONS.T; phase=projection[:,:,None]*FREQUENCIES[None,None,:]; feature=np.concatenate([np.cos(phase).reshape(len(state),-1),np.sin(phase).reshape(len(state),-1)],axis=1); require(feature.shape[1]==FEATURE_N and np.isfinite(feature).all(),"V214 ECF feature"); return feature

def ecf_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True); processor=numeric.RobustNumericState().fit(ordered); ts,ta=processor.transform(ordered); xs,xa=processor.transform(target); tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]); xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]); group_path,y,grouping=equal_event_group_path(ordered,tp); scale=TrainChannelScale().fit(group_path); train_path=scale.transform(group_path); target_path=scale.transform(xp); train_state=np.median(train_path,axis=2); target_state=np.median(target_path,axis=2); train_feature=ecf_features(train_state); target_feature=ecf_features(target_state)
    mean_down=train_feature[y==0].mean(axis=0); mean_up=train_feature[y==1].mean(axis=0); train_score=((train_feature-mean_down)**2).mean(axis=1)-((train_feature-mean_up)**2).mean(axis=1); target_score=((target_feature-mean_down)**2).mean(axis=1)-((target_feature-mean_up)**2).mean(axis=1); temperature=max(float(np.quantile(train_score,.75)-np.quantile(train_score,.25)),TEMP_FLOOR); prob=np.clip(1/(1+np.exp(-np.clip(target_score/temperature,-30,30))),1e-6,1-1e-6)
    geometry={"rows":len(train_feature),"feature_dimension":FEATURE_N,"fixed_cramer_wold_projection_bank":True,"class_conditional_empirical_characteristic_means":True,"complex_cosine_sine_coordinates":True,"direct_ecf_prototype_distance_contrast":True,"train_only_temperature":True,"no_discriminative_head":True,"target_transform_only":True,"model_fit_uses_target_rows":False,"target_labels_used":False,"label_independent_exact_transform":True,"local_differential_geometry_not_global_path_integral":False,"direction_sha256":array_sha256(DIRECTIONS),"mean_down_sha256":array_sha256(mean_down),"mean_up_sha256":array_sha256(mean_up),"temperature":temperature,"v84_random_rff_bayes":False,"v134_within_row_fourier_spectrum":False,"v147_sliced_wasserstein_quantiles":False,"v202_kernel_stein_score":False,"v211_hsic_gram_alignment":False,"v212_mca":False,"v213_dpp":False}
    return prob,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":geometry,"target_geometry":{**geometry,"rows":len(target_feature),"model_fit_uses_target_rows":False},"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"fixed_bounded_ecf_coordinates":True},"head":{"type":"none; direct class ECF prototype distance","temperature":temperature},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(prob.mean()),"std":float(prob.std()),"probability_sha256":array_sha256(prob)}}

_BASE_BOOTSTRAP=scaffold._BASE_BOOTSTRAP
def configure_base():
    core.VERSION=VERSION; core.HYPOTHESIS=HYPOTHESIS; core.STATIC_SPEC=STATIC_SPEC; core.DAG_FEATURE_DIMENSION=FEATURE_N; core.ARCHITECTURES=ARCHITECTURES; core.SEED=SEED; core.dag_direction=ecf_direction; core.choose_inner=choose_inner; base.VERSION=VERSION; base.HYPOTHESIS=HYPOTHESIS; base.STATIC_SPEC=STATIC_SPEC; base.FRENET_FEATURE_DIMENSION=FEATURE_N; base.ARCHITECTURES=ARCHITECTURES; base.SEED=SEED; base.frenet_direction=ecf_direction
def choose_inner(tr,va,ch):
    challenger,a=ecf_direction(tr,va); b=ch.prob.to_numpy(float); conf=ch.confidence_signal.to_numpy(float); high=bool_series(ch.high_conf).to_numpy(bool); bm=metric(va,b,conf,high); trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}]; g=a["train_geometry"]; ok=bool(g["fixed_cramer_wold_projection_bank"] and g["class_conditional_empirical_characteristic_means"] and g["direct_ecf_prototype_distance_contrast"] and g["train_only_temperature"] and g["no_discriminative_head"] and g["target_transform_only"] and not g["model_fit_uses_target_rows"] and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(b,challenger,ar["weight"]); m=metric(va,p,conf,high); auc=m["auc"]-bm["auc"]; ba=m["balanced_accuracy"]-bm["balanced_accuracy"]; net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"]; floor,d=base.supported_source_ba_delta(va,b,p); trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    selected=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP")); return {"selection_rule":"inner OOF V69 noop or fixed .25/.50 class Cramer-Wold ECF blend","baseline":bm,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"projection_frequency_temperature_blend_or_safety_micro_tuning":False},a
def run_nested(dev,ch,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()): d,e,a=core.run_nested(dev,ch,smoke,smoke_market)
    d=d.rename(columns={"v195_model":"v214_model"}); e=e.rename(columns={"sparse_dag_moment_prob":"cramer_wold_ecf_prob","v195_model":"v214_model"})
    for x in a:
        s=x["policy"]["selected"]; print(f"[V214 ECF] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws): configure_base(); r=_BASE_BOOTSTRAP(e,draws); r["method"]="paired market-fold-time block bootstrap over inner-locked V214 class ECF policy"; r["seed"]=SEED; r["standardized"]=True; return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base(); base.paired_nested_bootstrap=paired_nested_bootstrap; r=base.evaluate(ch,d,e,a,draws,smoke); c=r["material_checks"]; c.pop("multichannel_discrete_frenet_curvature_contract_verified"); contract=all(x[side]["train_geometry"]["class_conditional_empirical_characteristic_means"] and x[side]["train_geometry"]["direct_ecf_prototype_distance_contrast"] and x[side]["train_geometry"]["target_transform_only"] and not x[side]["model_fit_uses_target_rows"] and not x[side]["target_labels_used"] for x in a for side in ("inner_model_audit","outer_model_audit")); c["multichannel_class_conditional_cramer_wold_ecf_contract_verified"]=bool(contract); r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]: r["selected_frame"]=ch.copy(); r["selected_summary"]=r["baseline_summary"]; require(r["selected_frame"].equals(ch),"V214 fallback")
    return r
def material_gate(e): c=e["material_checks"]; g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v) for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate" if e["material_pass"] else "exact entire V69 DataFrame"}; require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key"); return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT) and out.resolve()==Path(CONTROLLER_OUTPUT).resolve() and out.resolve().parent==(ROOT/"research"/"staging"/"V214").resolve(),"binding"); required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"}; allowed=required|{"DATA_EPOCH_BINDING.json"}; existing={p.name for p in out.iterdir()}; require(required.issubset(existing) and not(existing-allowed),"conflict"); return {"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out); compact={k:v for k,v in e.items() if not k.endswith("_frame")}; gate=material_gate(e); selected=json.loads(json.dumps(clean(e["selected_summary"]))); selected["material_gate"]=gate; reports={"MODEL_COMPARISON.json":{"version":"V214","status":e["status"],"models":{"V69":e["baseline_summary"],"V214_DIAGNOSTIC":e["candidate_summary"],"V214_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V214","status":e["status"],"selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V214","status":e["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V214_CRAMER_WOLD_ECF_REPORT.json":{"status":e["status"],"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items(): atomic_json(p,out/n)
    atomic_csv(evidence,out/"V214_ECF_EVIDENCE.csv.gz"); atomic_csv(e["selected_frame"],out/"V214_FAIL_CLOSED_SELECTED_OOF.csv.gz"); files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name!="ARTIFACT_MANIFEST.json"}; atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json"); return {"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser(); m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); m.add_argument("--audit-only",action="store_true"); m.add_argument("--smoke-test",action="store_true"); m.add_argument("--support-probe",action="store_true"); m.add_argument("--full-run",action="store_true"); p.add_argument("--smoke-market",choices=("US","KR"),default="US"); p.add_argument("--output",type=Path); a=p.parse_args()
    if CONTROLLER_OUTPUT: require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"conflict"); a.full_run=True; a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None: a.output=DEFAULT_OUT
    if a.full_run: require(bool(CONTROLLER_OUTPUT) and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full"); return a
    return a
def main():
    a=parse_args(); bounded=bool(a.audit_only or a.smoke_test or a.support_probe); runtime=numeric.configure_bounded_runtime() if bounded else {"mode":"controller deterministic CPU class Cramer-Wold ECF","gpu_model_calls":0}; authority=verify_authority(); dev,ch=load_authorized()
    if a.audit_only: print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"dev_robustness_top_level_status":True,"nested_key":"selected.material_gate.nested_bootstrap","exact_fallback":True},"output_written":False}),indent=2)); return
    with threadpool_limits(limits=2 if bounded else None):
        d,e,aud=run_nested(dev,ch,bounded,a.smoke_market)
        if a.support_probe:
            x=aud[0]; non=[v for v in x["policy"]["trials"] if v["architecture"]]; best=max(non,key=lambda v:(v["eligible"],v["score"])); print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":x["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":x["outer_baseline"],"raw_outer_selected":x["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":x["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":x["inner_chronology"],"strict_outer_chronology":x["outer_chronology"],"output_written":False}),indent=2)); return
        ev=evaluate(ch,d,e,aud,250 if a.smoke_test else 2000,a.smoke_test)
    non=[v for v in aud[0]["policy"]["trials"] if v["architecture"]]; best=max(non,key=lambda v:(v["eligible"],v["score"])); s={"status":"SMOKE_OK" if a.smoke_test else ev["status"],"runtime":runtime,"selected_models":[x["policy"]["selected"]["name"] for x in aud],"outer_auc_delta":ev["outer_auc_delta"],"outer_ba_delta":ev["outer_ba_delta"],"outer_net_delta":ev["outer_net_delta"],"material_pass":ev["material_pass"],"fallback_is_exact_v69":not ev["material_pass"],"current_material_gate":material_gate(ev),"output_written":False}
    if a.smoke_test: s.update({"raw_inner_selected":aud[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":aud[0]["outer_baseline"],"raw_outer_selected":aud[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":aud[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]}); print(json.dumps(clean(s),indent=2)); return
    print(json.dumps(clean({**s,"output_written":True,"controller":write_outputs(a.output,authority,e,aud,ev)}),indent=2))
if __name__=="__main__": main()
