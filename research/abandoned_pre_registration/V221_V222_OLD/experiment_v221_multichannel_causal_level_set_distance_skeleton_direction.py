"""V221 strict-train level-set distance-skeleton direction challenger."""
from __future__ import annotations
import os
import ctypes
CPU_AFFINITY_MASK=0xC0000000
CONTROLLER_RESOURCE_MODE=bool(os.environ.get("MARKET_BIO_VERSION_OUTPUT"))
if os.name=="nt":
    _kernel32=ctypes.windll.kernel32;_kernel32.GetCurrentProcess.restype=ctypes.c_void_p;_kernel32.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t);_kernel32.SetProcessAffinityMask.restype=ctypes.c_bool;_kernel32.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t));_kernel32.GetProcessAffinityMask.restype=ctypes.c_bool;_process=_kernel32.GetCurrentProcess()
    if not CONTROLLER_RESOURCE_MODE and not _kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)):raise RuntimeError("V221 pre-import affinity")
if not CONTROLLER_RESOURCE_MODE:
    for _name in("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):os.environ[_name]="2"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
from contextlib import redirect_stdout,nullcontext
from io import StringIO
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
_RESOURCE_ENV_KEYS={"OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","CUDA_VISIBLE_DEVICES"}
if CONTROLLER_RESOURCE_MODE:
    _environment_type=type(os.environ);_saved_environment_setitem=_environment_type.__setitem__
    def _preserve_resource_environment(self,key,value):
        if str(key)in _RESOURCE_ENV_KEYS:return
        return _saved_environment_setitem(self,key,value)
    _environment_type.__setitem__=_preserve_resource_environment
    if os.name=="nt":
        _saved_set_affinity_api=_kernel32.SetProcessAffinityMask
        def _preserve_inherited_affinity(*_args):return True
        _preserve_inherited_affinity.argtypes=(ctypes.c_void_p,ctypes.c_size_t);_preserve_inherited_affinity.restype=ctypes.c_bool;_kernel32.SetProcessAffinityMask=_preserve_inherited_affinity
try:
    import experiment_v209_multichannel_causal_cross_horizon_regularized_cca_direction as scaffold
finally:
    if CONTROLLER_RESOURCE_MODE:
        _environment_type.__setitem__=_saved_environment_setitem
        if os.name=="nt":_kernel32.SetProcessAffinityMask=_saved_set_affinity_api
if not CONTROLLER_RESOURCE_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    if os.name=="nt" and not _kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)):raise RuntimeError("V221 post-scaffold affinity")

CONTROLLER_OUTPUT=os.environ.get("MARKET_BIO_VERSION_OUTPUT");base=scaffold.base;ROOT=scaffold.ROOT;VERSION=221;HYPOTHESIS="MULTICHANNEL_CAUSAL_LEVEL_SET_DISTANCE_SKELETON_DIRECTION_V1";DEFAULT_OUT=ROOT/"research"/"staging"/"V221"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES
SCAFFOLD_PATH=ROOT/"experiment_v209_multichannel_causal_cross_horizon_regularized_cca_direction.py";EXPECTED_SCAFFOLD_SHA256="60d45f438d985ac67688b19e970a741b2098a26c01e58b35585004931eba68ab"
LEVELS=np.array([-1.,-.5,0.,.5,1.]);REGION_FEATURE_N=8;REGION_N=5;FEATURE_DIMENSION=len(LEVELS)*2*REGION_N*REGION_FEATURE_N;SEED=22101;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"LEVEL_SET_DISTANCE_SKELETON_W0.25","weight":.25},{"name":"LEVEL_SET_DISTANCE_SKELETON_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
_ROW,_COL=np.indices((8,8));REGIONS=(np.ones((8,8),bool),_ROW<4,_ROW>=4,_COL<4,_COL>=4);_PIXELS=np.stack((_ROW.ravel(),_COL.ravel()),axis=1).astype(float);_EXTERIOR=np.unique(np.array([(-1,c)for c in range(-1,9)]+[(8,c)for c in range(-1,9)]+[(r,-1)for r in range(8)]+[(r,8)for r in range(8)],float),axis=0)
STATIC_SPEC={"path_shape":[8,8],"fixed_levels":LEVELS.tolist(),"filtrations":["sublevel","superlevel"],"distance":"exact Euclidean distance to within-grid complement plus immutable exterior boundary","medial_ridge":"positive 8-neighbor local maxima of distance field","regions":"whole plus four fixed axis halves","summaries_per_region":["active fraction","inside-distance mean","inside-distance q75","inside-distance max","ridge fraction","ridge-radius mean","ridge endpoint fraction","ridge junction fraction"],"feature_dimension":FEATURE_DIMENSION,"head":"equal-event-group class-balanced C=.5 ridge logistic","target_model_fit":False,"gpu_model_calls":0}

class RobustFeatureScale:
    def __init__(self):self.median=np.zeros(FEATURE_DIMENSION);self.scale=np.ones(FEATURE_DIMENSION)
    def fit(self,matrix):
        require(matrix.ndim==2 and matrix.shape[1]==FEATURE_DIMENSION,"V221 feature fit shape");self.median=np.median(matrix,axis=0);q25,q75=np.quantile(matrix,[.25,.75],axis=0);robust=(q75-q25)/1.349;standard=np.std(matrix,axis=0);self.scale=np.where(robust>1e-8,robust,np.where(standard>1e-8,standard,1.));return self
    def transform(self,matrix):require(matrix.ndim==2 and matrix.shape[1]==FEATURE_DIMENSION,"V221 feature transform shape");return np.clip((matrix-self.median)/self.scale,-7.,7.)

def observed_affinity(require_bounded=True):
    if os.name!="nt":return{"mask_hex":None,"cpus":None}
    pm=ctypes.c_size_t();sm=ctypes.c_size_t();require(_kernel32.GetProcessAffinityMask(_process,ctypes.byref(pm),ctypes.byref(sm)),"V221 affinity query");cpus=[i for i in range(64)if pm.value&(1<<i)]
    if require_bounded:require(pm.value==CPU_AFFINITY_MASK and cpus==[30,31],"V221 bounded affinity drift")
    return{"mask_hex":f"0x{pm.value:08X}","cpus":cpus}
def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V209 scaffold changed");authority=scaffold.verify_authority();authority["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V209 output"};authority["v221_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return authority

def exact_inside_distance(mask):
    background=np.concatenate((_PIXELS[~mask.ravel()],_EXTERIOR),axis=0);difference=_PIXELS[:,None,:]-background[None,:,:];distance=np.sqrt(np.min(np.sum(difference*difference,axis=2),axis=1)).reshape(8,8);distance[~mask]=0.;return distance
def medial_ridge(distance):
    ridge=distance>0
    for dr in(-1,0,1):
        for dc in(-1,0,1):
            if dr==0 and dc==0:continue
            shifted=np.full((8,8),-np.inf);rs=slice(max(0,dr),min(8,8+dr));cs=slice(max(0,dc),min(8,8+dc));ssr=slice(max(0,-dr),min(8,8-dr));ssc=slice(max(0,-dc),min(8,8-dc));shifted[rs,cs]=distance[ssr,ssc];ridge&=distance>=shifted-1e-12
    degree=np.zeros((8,8),int)
    for dr in(-1,0,1):
        for dc in(-1,0,1):
            if dr==0 and dc==0:continue
            rs=slice(max(0,dr),min(8,8+dr));cs=slice(max(0,dc),min(8,8+dc));ssr=slice(max(0,-dr),min(8,8-dr));ssc=slice(max(0,-dc),min(8,8-dc));degree[rs,cs]+=ridge[ssr,ssc]
    return ridge,degree
def region_summary(mask,distance,ridge,degree,region):
    active=mask&region;medial=ridge&region;region_n=float(region.sum());inside=distance[active];radius=distance[medial];endpoint=medial&(degree<=1);junction=medial&(degree>=3)
    return np.array([active.sum()/region_n,float(inside.mean())if len(inside)else 0.,float(np.quantile(inside,.75))if len(inside)else 0.,float(inside.max())if len(inside)else 0.,medial.sum()/region_n,float(radius.mean())if len(radius)else 0.,endpoint.sum()/region_n,junction.sum()/region_n])
def distance_skeleton_features(path):
    require(path.ndim==3 and path.shape[1:]==(8,8),"V221 path shape");rows=[];ridge_counts=[]
    for surface in path:
        summaries=[];event_ridge_n=0
        for level in LEVELS:
            for mask in(surface<=level,surface>=level):
                distance=exact_inside_distance(mask);ridge,degree=medial_ridge(distance);event_ridge_n+=int(ridge.sum())
                for region in REGIONS:summaries.append(region_summary(mask,distance,ridge,degree,region))
        rows.append(np.concatenate(summaries));ridge_counts.append(event_ridge_n)
    feature=np.stack(rows);require(feature.shape==(len(path),FEATURE_DIMENSION)and np.isfinite(feature).all(),"V221 feature")
    audit={"rows":len(path),"feature_dimension":FEATURE_DIMENSION,"fixed_sub_and_super_level_sets":True,"exact_euclidean_distance_to_complement_and_exterior":True,"deterministic_eight_neighbor_medial_ridges":True,"ridge_endpoint_and_junction_summaries":True,"whole_and_four_fixed_regions":True,"ridge_count_min":int(min(ridge_counts)),"ridge_count_max":int(max(ridge_counts)),"level_sha256":array_sha256(LEVELS),"label_independent_exact_transform":True,"local_differential_geometry_not_global_path_integral":True,"model_fit_uses_target_rows":False,"target_labels_used":False,"v167_euler_curve":False,"v174_morphological_granulometry":False,"v188_persistent_homology":False,"v190_discrete_morse":False,"v197_lacunarity":False,"v219_gray_level_size_zone":False,"v220_ngtdm":False};return feature,audit
def distance_skeleton_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);train_state,train_transform=processor.transform(ordered);target_state,target_transform=processor.transform(target);train_path,train_path_audit=observed_past_to_recent_path(train_state[:,:len(FEATURES)]);target_path,target_path_audit=observed_past_to_recent_path(target_state[:,:len(FEATURES)]);group_path,group_target,grouping=equal_event_group_path(ordered,train_path);channel_scale=TrainChannelScale().fit(group_path);group_path=channel_scale.transform(group_path);target_path=channel_scale.transform(target_path);train_feature,train_geometry=distance_skeleton_features(group_path);target_feature,target_geometry=distance_skeleton_features(target_path);feature_scale=RobustFeatureScale().fit(train_feature);train_design=feature_scale.transform(train_feature);target_design=feature_scale.transform(target_feature);class_n=np.bincount(group_target,minlength=2).astype(float);require(np.all(class_n>0),"V221 class support");sample_weight=np.where(group_target==1,.5/class_n[1],.5/class_n[0]);head=LogisticRegression(C=.5,penalty="l2",solver="liblinear",max_iter=300,random_state=SEED);head.fit(train_design,group_target,sample_weight=sample_weight);probability=np.clip(head.predict_proba(target_design)[:,1],1e-6,1-1e-6)
    return probability,{"train_n":len(ordered),"train_event_group_n":len(group_target),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":train_transform,"target_transform":target_transform,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":train_path_audit,"target_path":target_path_audit,"train_geometry":train_geometry,"target_geometry":target_geometry,"channel_scaling":{"train_only":True,"median_sha256":array_sha256(channel_scale.median),"scale_sha256":array_sha256(channel_scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"median_sha256":array_sha256(feature_scale.median),"scale_sha256":array_sha256(feature_scale.scale),"clip":7.},"head":{"type":"fixed equal-event-group/class-balanced ridge logistic","C":.5,"solver":"liblinear","coefficient_sha256":array_sha256(head.coef_),"coefficient_l2":float(np.linalg.norm(head.coef_)),"intercept":head.intercept_.tolist()},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"diffusion_gmrf_riesz_tropical_glszm_or_ngtdm":False,"gpu_model_calls":0,"prediction":{"mean":float(probability.mean()),"std":float(probability.std()),"probability_sha256":array_sha256(probability)}}

_BASE_BOOTSTRAP=base.paired_nested_bootstrap
def configure_base():base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=FEATURE_DIMENSION;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=distance_skeleton_direction;base.choose_inner=choose_inner
def choose_inner(train,valid,champion):
    challenger,audit=distance_skeleton_direction(train,valid);baseline=champion.prob.to_numpy(float);confidence=champion.confidence_signal.to_numpy(float);high=bool_series(champion.high_conf).to_numpy(bool);baseline_metric=metric(valid,baseline,confidence,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}];geometry=audit["train_geometry"];model_ok=bool(geometry["fixed_sub_and_super_level_sets"]and geometry["exact_euclidean_distance_to_complement_and_exterior"]and geometry["deterministic_eight_neighbor_medial_ridges"]and geometry["ridge_endpoint_and_junction_summaries"]and not geometry["model_fit_uses_target_rows"]and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]);measured=metric(valid,probability,confidence,high);auc_delta=measured["auc"]-baseline_metric["auc"];ba_delta=measured["balanced_accuracy"]-baseline_metric["balanced_accuracy"];net_delta=measured["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"];source_floor,source_delta=base.supported_source_ba_delta(valid,baseline,probability);trials.append({"name":architecture["name"],"architecture":architecture,"metrics":measured,"auc_delta":auc_delta,"balanced_accuracy_delta":ba_delta,"all_trade_net_delta":net_delta,"worst_supported_source_ba_delta":source_floor,"supported_source_ba_delta":source_delta,"model_eligible":model_ok,"eligible":bool(model_ok and ba_delta>=-.005 and net_delta>=-.0005 and source_floor>=-.01),"score":2*measured["auc"]+measured["balanced_accuracy"]})
    selected=max([trial for trial in trials if trial["eligible"]],key=lambda trial:(trial["score"],trial["auc_delta"],trial["all_trade_net_delta"],trial["name"]=="V69_NOOP"));return{"selection_rule":"inner OOF V69 noop or fixed .25/.50 level-set distance-skeleton blend","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"level_region_distance_ridge_head_blend_or_safety_micro_tuning":False},audit
def run_nested(dev,champion,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):diagnostic,evidence,audits=base.run_nested(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v146_model":"v221_model"});evidence=evidence.rename(columns={"frenet_prob":"distance_skeleton_prob","v146_model":"v221_model"})
    for audit in audits:selected=audit["policy"]["selected"];print(f"[V221 DISTANCE SKELETON] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}")
    return diagnostic,evidence,audits
def paired_nested_bootstrap(evidence,draws):configure_base();result=_BASE_BOOTSTRAP(evidence,draws);result["method"]="paired market-fold-time block bootstrap over inner-locked V221 level-set distance-skeleton policy";result["seed"]=SEED;result["standardized"]=True;return result
def evaluate(champion,diagnostic,evidence,audits,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;result=base.evaluate(champion,diagnostic,evidence,audits,draws,smoke);checks=result["material_checks"];checks.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(audit[side]["train_geometry"]["fixed_sub_and_super_level_sets"]and audit[side]["train_geometry"]["exact_euclidean_distance_to_complement_and_exterior"]and audit[side]["train_geometry"]["deterministic_eight_neighbor_medial_ridges"]and audit[side]["train_geometry"]["ridge_endpoint_and_junction_summaries"]and not audit[side]["train_geometry"]["model_fit_uses_target_rows"]and not audit[side]["target_labels_used"]for audit in audits for side in("inner_model_audit","outer_model_audit"));checks["multichannel_causal_level_set_distance_skeleton_contract_verified"]=bool(contract);result["material_pass"]=bool(not smoke and all(checks.values()))
    if not result["material_pass"]:result["selected_frame"]=champion.copy();result["selected_summary"]=result["baseline_summary"];require(result["selected_frame"].equals(champion),"V221 exact fallback")
    return result
def material_gate(evaluation):checks=evaluation["material_checks"];gate={"contract":HYPOTHESIS,"checks":checks,"passed":sum(bool(value)for value in checks.values()),"total":len(checks),"material_pass":bool(evaluation["material_pass"]),"nested_bootstrap":evaluation["nested_bootstrap"],"native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if evaluation["material_pass"]else"exact entire V69 DataFrame"};require(gate["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V221 nested key");return gate
def enforce_output_conflict_fail_closed(out):require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V221").resolve(),"V221 binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={path.name for path in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"V221 conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,evaluation):
    conflict=enforce_output_conflict_fail_closed(out);compact={key:value for key,value in evaluation.items()if not key.endswith("_frame")};gate=material_gate(evaluation);selected=json.loads(json.dumps(clean(evaluation["selected_summary"])));selected["material_gate"]=gate;reports={"MODEL_COMPARISON.json":{"version":"V221","status":evaluation["status"],"models":{"V69":evaluation["baseline_summary"],"V221_DIAGNOSTIC":evaluation["candidate_summary"],"V221_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V221","status":evaluation["status"],"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V221","status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V221_LEVEL_SET_DISTANCE_SKELETON_REPORT.json":{"status":evaluation["status"],"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":evaluation["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":evaluation["status"],"priority":895,"conflict":conflict}}
    for name,payload in reports.items():atomic_json(payload,out/name)
    atomic_csv(evidence,out/"V221_LEVEL_SET_DISTANCE_SKELETON_EVIDENCE.csv.gz");atomic_csv(evaluation["selected_frame"],out/"V221_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={path.name:{"sha256":sha256(path),"bytes":path.stat().st_size}for path in out.iterdir()if path.is_file()and path.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));mode.add_argument("--audit-only",action="store_true");mode.add_argument("--smoke-test",action="store_true");mode.add_argument("--support-probe",action="store_true");mode.add_argument("--full-run",action="store_true");parser.add_argument("--smoke-market",choices=("US","KR"),default="US");parser.add_argument("--output",type=Path);args=parser.parse_args()
    if CONTROLLER_OUTPUT:require(not(args.audit_only or args.smoke_test or args.support_probe or args.full_run or args.output),"V221 controller conflict");args.full_run=True;args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None:args.output=DEFAULT_OUT
    if args.full_run:require(bool(CONTROLLER_OUTPUT)and args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V221 direct full")
    return args
def main():
    args=parse_args();bounded=bool(args.audit_only or args.smoke_test or args.support_probe);require(bounded or CONTROLLER_RESOURCE_MODE,"V221 unbound full resource mode");runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller inherited full resources level-set distance skeleton","gpu_model_calls":0,"numeric_thread_cap":None,"resource_mutation":False}
    if bounded:
        if os.name=="nt":require(_kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)),"V221 bounded runtime affinity")
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    runtime["observed_affinity"]=observed_affinity(bounded);runtime["gpu_visible"]=os.environ.get("CUDA_VISIBLE_DEVICES");runtime["gpu_model_calls"]=0;authority=verify_authority();dev,champion=load_authorized()
    if args.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"dev_robustness_top_level_status":True,"nested_key":"selected.material_gate.nested_bootstrap","native_robustness":True,"exact_fallback":True},"output_written":False}),indent=2));return
    pool_context=threadpool_limits(limits=2)if bounded else nullcontext()
    with pool_context:
        diagnostic,evidence,audits=run_nested(dev,champion,bounded,args.smoke_market)
        if args.support_probe:
            audit=audits[0];non_noop=[trial for trial in audit["policy"]["trials"]if trial["architecture"]];best=max(non_noop,key=lambda trial:(trial["eligible"],trial["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"output_written":False}),indent=2));return
        evaluation=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,args.smoke_test)
    non_noop=[trial for trial in audits[0]["policy"]["trials"]if trial["architecture"]];best=max(non_noop,key=lambda trial:(trial["eligible"],trial["score"]));summary={"status":"SMOKE_OK"if args.smoke_test else evaluation["status"],"runtime":runtime,"selected_models":[audit["policy"]["selected"]["name"]for audit in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test:summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(summary),indent=2));return
    print(json.dumps(clean({**summary,"output_written":True,"controller":write_outputs(args.output,authority,evidence,audits,evaluation)}),indent=2))
if __name__=="__main__":main()
