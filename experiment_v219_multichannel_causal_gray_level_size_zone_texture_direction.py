"""V219 fixed gray-level size-zone texture direction challenger."""
from __future__ import annotations
import os
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_n] = "2"
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import experiment_v217_multichannel_causal_monogenic_riesz_phase_direction as scaffold

core=scaffold.core; base=scaffold.base; ROOT=scaffold.ROOT; VERSION=219
HYPOTHESIS="MULTICHANNEL_CAUSAL_GRAY_LEVEL_SIZE_ZONE_TEXTURE_DIRECTION_V1"
DEFAULT_OUT=ROOT/"research"/"staging"/"V219"/"LOCAL_FORBIDDEN"; FEATURES=scaffold.FEATURES
SCAFFOLD_PATH=ROOT/"experiment_v217_multichannel_causal_monogenic_riesz_phase_direction.py"; EXPECTED_SCAFFOLD_SHA256="d6b08b8e4ea5886c9c9c1246e39cfb85ba21bdc0a9462bbdcc4a54656ab8567a"
LEVEL_THRESHOLDS=np.array([-1.0,-0.5,0.5,1.0]); LEVEL_N=5; REGION_N=5; REGION_FEATURE_N=24; FEATURE_N=120; SEED=21901
ARCHITECTURES=({"name":"GRAY_LEVEL_SIZE_ZONE_W0.25","weight":.25},{"name":"GRAY_LEVEL_SIZE_ZONE_W0.50","weight":.50})
require=scaffold.require; sha256=scaffold.sha256; array_sha256=scaffold.array_sha256; clean=scaffold.clean; bool_series=scaffold.bool_series; atomic_json=scaffold.atomic_json; atomic_csv=scaffold.atomic_csv
load_authorized=scaffold.load_authorized; metric=scaffold.metric; controller=scaffold.controller; numeric=scaffold.numeric; observed_past_to_recent_path=scaffold.observed_past_to_recent_path; equal_event_group_path=scaffold.equal_event_group_path; TrainChannelScale=scaffold.TrainChannelScale; blend_probability=scaffold.blend_probability
REGIONS=(np.ones((8,8),bool),np.indices((8,8))[0]<4,np.indices((8,8))[0]>=4,np.indices((8,8))[1]<4,np.indices((8,8))[1]>=4)
STATIC_SPEC={"feature_dimension":FEATURE_N,"surface":"strict-train scaled continuous 8x8 causal surface quantized at fixed [-1,-.5,.5,1]","topology":"exact four-neighbor connected same-level 2D zones","regions":"whole plus four half-surface regions","matrix":"5 gray levels by zone sizes 1..64","summaries_per_region":REGION_FEATURE_N,"head":"equal-event-group class-balanced C=.5 ridge logistic","target_model_fit":False}

class RobustFeatureScale:
    def __init__(self):self.median=np.zeros(FEATURE_N);self.scale=np.ones(FEATURE_N)
    def fit(self,matrix):
        require(matrix.ndim==2 and matrix.shape[1]==FEATURE_N,"V219 feature fit shape");self.median=np.median(matrix,axis=0);q25,q75=np.quantile(matrix,[.25,.75],axis=0);robust=(q75-q25)/1.349;standard=np.std(matrix,axis=0);self.scale=np.where(robust>1e-8,robust,np.where(standard>1e-8,standard,1.));return self
    def transform(self,matrix):require(matrix.ndim==2 and matrix.shape[1]==FEATURE_N,"V219 feature transform shape");return np.clip((matrix-self.median)/self.scale,-7.,7.)

def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V217 scaffold changed"); authority=scaffold.verify_authority(); authority["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V217 output"}; authority["v219_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False}; return authority

def zone_matrix(level_surface,mask):
    matrix=np.zeros((LEVEL_N,64),float); visited=np.zeros((8,8),bool)
    for row in range(8):
        for col in range(8):
            if not mask[row,col] or visited[row,col]: continue
            level=int(level_surface[row,col]); stack=[(row,col)]; visited[row,col]=True; size=0
            while stack:
                current_row,current_col=stack.pop(); size+=1
                for next_row,next_col in ((current_row-1,current_col),(current_row+1,current_col),(current_row,current_col-1),(current_row,current_col+1)):
                    if 0<=next_row<8 and 0<=next_col<8 and mask[next_row,next_col] and not visited[next_row,next_col] and int(level_surface[next_row,next_col])==level:
                        visited[next_row,next_col]=True; stack.append((next_row,next_col))
            matrix[level,size-1]+=1.0
    return matrix

def zone_summaries(matrix,pixel_n):
    total=float(matrix.sum()); require(total>0,"V219 empty zone matrix"); probability=matrix/total
    gray=np.arange(1,LEVEL_N+1,dtype=float)[:,None]; size=np.arange(1,65,dtype=float)[None,:]
    gray_mass=probability.sum(axis=1); size_mass=probability.sum(axis=0); gray_mean=float((probability*gray).sum()); size_mean=float((probability*size).sum())
    nonzero=probability[probability>0]; gray_center=gray-gray_mean; size_center=size-size_mean; gray_sd=np.sqrt(float((probability*gray_center**2).sum())); size_sd=np.sqrt(float((probability*size_center**2).sum())); correlation=float((probability*gray_center*size_center).sum()/max(gray_sd*size_sd,1e-12))
    return np.array([
        float((probability/size**2).sum()),float((probability*size**2).sum()),float((gray_mass**2).sum()),float((gray_mass**2).sum()/total),
        float((size_mass**2).sum()),float((size_mass**2).sum()/total),total/float(pixel_n),float((probability/gray**2).sum()),float((probability*gray**2).sum()),
        float((probability/(gray**2*size**2)).sum()),float((probability*gray**2/size**2).sum()),float((probability*size**2/gray**2).sum()),float((probability*gray**2*size**2).sum()),
        gray_mean,float((probability*gray_center**2).sum()),size_mean,float((probability*size_center**2).sum()),float(-(nonzero*np.log(nonzero)).sum()),
        float(np.max(np.where(size_mass>0,np.arange(1,65),0))/pixel_n),float(size_mass[0]),float(size_mass[1:].sum()),float(np.count_nonzero(gray_mass)/LEVEL_N),float(probability.max()),correlation
    ])

def glszm_features(path):
    require(path.ndim==3 and path.shape[1:]==(8,8),"V219 path shape"); levels=np.digitize(path,LEVEL_THRESHOLDS).astype(np.int8); rows=[]; total_zone_n=[]
    for surface in levels:
        summaries=[]; zone_n=0
        for mask in REGIONS:
            matrix=zone_matrix(surface,mask); summaries.append(zone_summaries(matrix,int(mask.sum()))); zone_n+=int(matrix.sum())
        rows.append(np.concatenate(summaries)); total_zone_n.append(zone_n)
    feature=np.stack(rows); require(feature.shape==(len(path),FEATURE_N) and np.isfinite(feature).all(),"V219 GLSZM feature")
    audit={"rows":len(path),"feature_dimension":FEATURE_N,"fixed_five_gray_levels":True,"exact_four_neighbor_connected_components":True,"gray_level_by_zone_size_matrices":True,"whole_and_four_fixed_regions":True,"fixed_glszm_summary_family":True,"zone_count_min":int(np.min(total_zone_n)),"zone_count_max":int(np.max(total_zone_n)),"threshold_sha256":array_sha256(LEVEL_THRESHOLDS),"label_independent_exact_transform":True,"local_differential_geometry_not_global_path_integral":True,"model_fit_uses_target_rows":False,"target_labels_used":False,"v174_morphological_granulometry":False,"v181_directional_glcm":False,"v184_local_ternary_pattern":False,"v187_gray_level_run_length":False,"v188_persistent_homology":False,"v216_class_gmrf_precision":False,"v217_monogenic_riesz":False,"v218_tropical_medoid":False}
    return feature,audit

def zone_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True); processor=numeric.RobustNumericState().fit(ordered); train_state,train_transform=processor.transform(ordered); target_state,target_transform=processor.transform(target); train_path,train_path_audit=observed_past_to_recent_path(train_state[:,:len(FEATURES)]); target_path,target_path_audit=observed_past_to_recent_path(target_state[:,:len(FEATURES)]); group_path,group_target,grouping=equal_event_group_path(ordered,train_path); channel_scale=TrainChannelScale().fit(group_path); group_path=channel_scale.transform(group_path); target_path=channel_scale.transform(target_path); train_feature,train_geometry=glszm_features(group_path); target_feature,target_geometry=glszm_features(target_path); feature_scale=RobustFeatureScale().fit(train_feature); train_design=feature_scale.transform(train_feature); target_design=feature_scale.transform(target_feature); class_n=np.bincount(group_target,minlength=2).astype(float); require(np.all(class_n>0),"V219 class support"); sample_weight=np.where(group_target==1,.5/class_n[1],.5/class_n[0]); head=LogisticRegression(C=.5,penalty="l2",solver="liblinear",max_iter=300,random_state=SEED); head.fit(train_design,group_target,sample_weight=sample_weight); probability=np.clip(head.predict_proba(target_design)[:,1],1e-6,1-1e-6)
    return probability,{"train_n":len(ordered),"train_event_group_n":len(group_target),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":train_transform,"target_transform":target_transform,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":train_path_audit,"target_path":target_path_audit,"train_geometry":train_geometry,"target_geometry":target_geometry,"channel_scaling":{"train_only":True,"median_sha256":array_sha256(channel_scale.median),"scale_sha256":array_sha256(channel_scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"median_sha256":array_sha256(feature_scale.median),"scale_sha256":array_sha256(feature_scale.scale),"clip":7.},"head":{"type":"fixed equal-event-group/class-balanced ridge logistic","C":.5,"solver":"liblinear","coefficient_sha256":array_sha256(head.coef_),"coefficient_l2":float(np.linalg.norm(head.coef_)),"intercept":head.intercept_.tolist()},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(probability.mean()),"std":float(probability.std()),"probability_sha256":array_sha256(probability)}}

_BASE_BOOTSTRAP=scaffold._BASE_BOOTSTRAP
def configure_base():
    core.VERSION=VERSION;core.HYPOTHESIS=HYPOTHESIS;core.STATIC_SPEC=STATIC_SPEC;core.DAG_FEATURE_DIMENSION=FEATURE_N;core.ARCHITECTURES=ARCHITECTURES;core.SEED=SEED;core.dag_direction=zone_direction;core.choose_inner=choose_inner;base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=FEATURE_N;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=zone_direction
def choose_inner(train,valid,champion):
    challenger,audit=zone_direction(train,valid);baseline=champion.prob.to_numpy(float);confidence=champion.confidence_signal.to_numpy(float);high=bool_series(champion.high_conf).to_numpy(bool);baseline_metric=metric(valid,baseline,confidence,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}];geometry=audit["train_geometry"];model_ok=bool(geometry["fixed_five_gray_levels"] and geometry["exact_four_neighbor_connected_components"] and geometry["gray_level_by_zone_size_matrices"] and geometry["fixed_glszm_summary_family"] and not geometry["model_fit_uses_target_rows"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]);measured=metric(valid,probability,confidence,high);auc_delta=measured["auc"]-baseline_metric["auc"];ba_delta=measured["balanced_accuracy"]-baseline_metric["balanced_accuracy"];net_delta=measured["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"];source_floor,source_delta=base.supported_source_ba_delta(valid,baseline,probability);trials.append({"name":architecture["name"],"architecture":architecture,"metrics":measured,"auc_delta":auc_delta,"balanced_accuracy_delta":ba_delta,"all_trade_net_delta":net_delta,"worst_supported_source_ba_delta":source_floor,"supported_source_ba_delta":source_delta,"model_eligible":model_ok,"eligible":bool(model_ok and ba_delta>=-.005 and net_delta>=-.0005 and source_floor>=-.01),"score":2*measured["auc"]+measured["balanced_accuracy"]})
    selected=max([trial for trial in trials if trial["eligible"]],key=lambda trial:(trial["score"],trial["auc_delta"],trial["all_trade_net_delta"],trial["name"]=="V69_NOOP"));return {"selection_rule":"inner OOF V69 noop or fixed .25/.50 gray-level size-zone blend","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"threshold_region_connectivity_summary_head_blend_or_safety_micro_tuning":False},audit
def run_nested(dev,champion,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()): diagnostic,evidence,audits=core.run_nested(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v195_model":"v219_model"});evidence=evidence.rename(columns={"sparse_dag_moment_prob":"gray_level_size_zone_prob","v195_model":"v219_model"})
    for audit in audits:
        selected=audit["policy"]["selected"];print(f"[V219 GLSZM] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}")
    return diagnostic,evidence,audits
def paired_nested_bootstrap(evidence,draws): configure_base();result=_BASE_BOOTSTRAP(evidence,draws);result["method"]="paired market-fold-time block bootstrap over inner-locked V219 gray-level size-zone policy";result["seed"]=SEED;result["standardized"]=True;return result
def evaluate(champion,diagnostic,evidence,audits,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;result=base.evaluate(champion,diagnostic,evidence,audits,draws,smoke);checks=result["material_checks"];checks.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(audit[side]["train_geometry"]["fixed_five_gray_levels"] and audit[side]["train_geometry"]["exact_four_neighbor_connected_components"] and audit[side]["train_geometry"]["gray_level_by_zone_size_matrices"] and audit[side]["train_geometry"]["fixed_glszm_summary_family"] and not audit[side]["train_geometry"]["model_fit_uses_target_rows"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit","outer_model_audit"));checks["multichannel_causal_gray_level_size_zone_texture_contract_verified"]=bool(contract);result["material_pass"]=bool(not smoke and all(checks.values()))
    if not result["material_pass"]:result["selected_frame"]=champion.copy();result["selected_summary"]=result["baseline_summary"];require(result["selected_frame"].equals(champion),"V219 exact fallback")
    return result
def material_gate(evaluation):
    checks=evaluation["material_checks"];gate={"contract":HYPOTHESIS,"checks":checks,"passed":sum(bool(value) for value in checks.values()),"total":len(checks),"material_pass":bool(evaluation["material_pass"]),"nested_bootstrap":evaluation["nested_bootstrap"],"native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"};require(gate["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V219 nested key");return gate
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT) and out.resolve()==Path(CONTROLLER_OUTPUT).resolve() and out.resolve().parent==(ROOT/"research"/"staging"/"V219").resolve(),"V219 binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={path.name for path in out.iterdir()};require(required.issubset(existing) and not(existing-allowed),"V219 conflict");return {"controller_bound":True}
def write_outputs(out,authority,evidence,audits,evaluation):
    conflict=enforce_output_conflict_fail_closed(out);compact={key:value for key,value in evaluation.items() if not key.endswith("_frame")};gate=material_gate(evaluation);selected=json.loads(json.dumps(clean(evaluation["selected_summary"])));selected["material_gate"]=gate;reports={"MODEL_COMPARISON.json":{"version":"V219","status":evaluation["status"],"models":{"V69":evaluation["baseline_summary"],"V219_DIAGNOSTIC":evaluation["candidate_summary"],"V219_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V219","status":evaluation["status"],"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V219","status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V219_GRAY_LEVEL_SIZE_ZONE_REPORT.json":{"status":evaluation["status"],"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":evaluation["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":evaluation["status"],"priority":885,"conflict":conflict}}
    for name,payload in reports.items():atomic_json(payload,out/name)
    atomic_csv(evidence,out/"V219_GRAY_LEVEL_SIZE_ZONE_EVIDENCE.csv.gz");atomic_csv(evaluation["selected_frame"],out/"V219_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={path.name:{"sha256":sha256(path),"bytes":path.stat().st_size} for path in out.iterdir() if path.is_file() and path.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return {"artifact_count":len(files)+1}
def parse_args():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));mode.add_argument("--audit-only",action="store_true");mode.add_argument("--smoke-test",action="store_true");mode.add_argument("--support-probe",action="store_true");mode.add_argument("--full-run",action="store_true");parser.add_argument("--smoke-market",choices=("US","KR"),default="US");parser.add_argument("--output",type=Path);args=parser.parse_args()
    if CONTROLLER_OUTPUT:require(not(args.audit_only or args.smoke_test or args.support_probe or args.full_run or args.output),"V219 controller conflict");args.full_run=True;args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None:args.output=DEFAULT_OUT
    if args.full_run:require(bool(CONTROLLER_OUTPUT) and args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V219 direct full")
    return args
def main():
    args=parse_args();bounded=bool(args.audit_only or args.smoke_test or args.support_probe);runtime=numeric.configure_bounded_runtime() if bounded else {"mode":"controller inherited full resources; CPU gray-level size-zone","gpu_model_calls":0};authority=verify_authority();dev,champion=load_authorized()
    if args.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"dev_robustness_top_level_status":True,"nested_key":"selected.material_gate.nested_bootstrap","exact_fallback":True},"output_written":False}),indent=2));return
    with threadpool_limits(limits=2 if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,bounded,args.smoke_market)
        if args.support_probe:
            audit=audits[0];non_noop=[trial for trial in audit["policy"]["trials"] if trial["architecture"]];best=max(non_noop,key=lambda trial:(trial["eligible"],trial["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"output_written":False}),indent=2));return
        evaluation=evaluate(champion,diagnostic,evidence,audits,250 if args.smoke_test else 2000,args.smoke_test)
    non_noop=[trial for trial in audits[0]["policy"]["trials"] if trial["architecture"]];best=max(non_noop,key=lambda trial:(trial["eligible"],trial["score"]));summary={"status":"SMOKE_OK" if args.smoke_test else evaluation["status"],"runtime":runtime,"selected_models":[audit["policy"]["selected"]["name"] for audit in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test:summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(summary),indent=2));return
    print(json.dumps(clean({**summary,"output_written":True,"controller":write_outputs(args.output,authority,evidence,audits,evaluation)}),indent=2))
if __name__=="__main__":main()
