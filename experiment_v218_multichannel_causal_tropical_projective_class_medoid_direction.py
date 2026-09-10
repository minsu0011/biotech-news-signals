"""V218 strict-train tropical-projective class-medoid direction."""
from __future__ import annotations
import os
import ctypes
CPU_AFFINITY_MASK=0xC0000000
CONTROLLER_RESOURCE_MODE=bool(os.environ.get("MARKET_BIO_VERSION_OUTPUT"))
if os.name=="nt":
    _kernel32=ctypes.windll.kernel32;_kernel32.GetCurrentProcess.restype=ctypes.c_void_p;_kernel32.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t);_kernel32.SetProcessAffinityMask.restype=ctypes.c_bool;_kernel32.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t));_kernel32.GetProcessAffinityMask.restype=ctypes.c_bool;_process=_kernel32.GetCurrentProcess()
    if not CONTROLLER_RESOURCE_MODE and not _kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)):raise RuntimeError("V218 pre-import affinity")
if not CONTROLLER_RESOURCE_MODE:
    for _name in("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):os.environ[_name]="2"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
from contextlib import redirect_stdout,nullcontext
from io import StringIO
import json
from pathlib import Path
import numpy as np
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
    if os.name=="nt" and not _kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)):raise RuntimeError("V218 post-scaffold affinity")

CONTROLLER_OUTPUT=os.environ.get("MARKET_BIO_VERSION_OUTPUT");base=scaffold.base;ROOT=scaffold.ROOT;VERSION=218;HYPOTHESIS="MULTICHANNEL_CAUSAL_TROPICAL_PROJECTIVE_CLASS_MEDOID_DIRECTION_V1";DEFAULT_OUT=ROOT/"research"/"staging"/"V218"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v209_multichannel_causal_cross_horizon_regularized_cca_direction.py";EXPECTED_SCAFFOLD_SHA256="60d45f438d985ac67688b19e970a741b2098a26c01e58b35585004931eba68ab"
SITE_N=64;FEATURE_DIMENSION=64;MEDOID_CHUNK=16;SCORE_CLIP=30.;SEED=21801;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"TROPICAL_PROJECTIVE_MEDOID_W0.25","weight":.25},{"name":"TROPICAL_PROJECTIVE_MEDOID_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
STATIC_SPEC={"path_shape":[8,8],"continuous_state_dimension":SITE_N,"gauge":"subtract per-row coordinate mean","metric":"Hilbert tropical projective max coordinate difference minus min coordinate difference","prototype":"exact deterministic equal-event-group medoid per class","score":"DOWN distance minus UP distance divided by strict-train robust IQR temperature","feature_dimension":FEATURE_DIMENSION,"head":"none; direct class-medoid distance contrast sigmoid","target_model_fit":False,"gpu_model_calls":0}

def observed_affinity(require_bounded=True):
    if os.name!="nt":return{"mask_hex":None,"cpus":None}
    pm=ctypes.c_size_t();sm=ctypes.c_size_t();require(_kernel32.GetProcessAffinityMask(_process,ctypes.byref(pm),ctypes.byref(sm)),"V218 affinity query");cpus=[i for i in range(64)if pm.value&(1<<i)]
    if require_bounded:require(pm.value==CPU_AFFINITY_MASK and cpus==[30,31],"V218 bounded affinity drift")
    return{"mask_hex":f"0x{pm.value:08X}","cpus":cpus}
def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V209 scaffold changed");a=scaffold.verify_authority();a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V209 output"};a["v218_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return a

def gauge(path):
    flat=path.reshape(len(path),SITE_N);centered=flat-flat.mean(1,keepdims=True);require(centered.shape==(len(path),SITE_N)and np.isfinite(centered).all(),"V218 gauge");return centered
def projective_distance(x,y):
    difference=x[:,None,:]-y[None,:,:];return difference.max(2)-difference.min(2)
def exact_medoid(x):
    require(len(x)>0,"V218 class support");objective=np.empty(len(x),dtype=float)
    for start in range(0,len(x),MEDOID_CHUNK):objective[start:start+MEDOID_CHUNK]=projective_distance(x[start:start+MEDOID_CHUNK],x).mean(1)
    index=int(np.argmin(objective));return x[index].copy(),index,objective
def fit_tropical(path,y):
    x=gauge(path);prototypes=[];indices=[];objectives=[]
    for label in(0,1):
        subset=x[y==label];prototype,index,objective=exact_medoid(subset);prototypes.append(prototype);indices.append(index);objectives.append(objective)
    prototypes=np.asarray(prototypes);train_score=projective_distance(x,prototypes)[:,0]-projective_distance(x,prototypes)[:,1];q25,q75=np.quantile(train_score,[.25,.75]);temperature=(q75-q25)/1.349
    if not np.isfinite(temperature)or temperature<=1e-8:temperature=float(np.std(train_score))
    if not np.isfinite(temperature)or temperature<=1e-8:temperature=1.
    model={"prototypes":prototypes,"temperature":float(temperature)};audit={"rows":len(path),"feature_dimension":FEATURE_DIMENSION,"class_counts":np.bincount(y,minlength=2).tolist(),"class_medoid_local_indices":indices,"class_medoid_objective_minima":[float(v.min())for v in objectives],"prototype_sha256":array_sha256(prototypes),"train_score_sha256":array_sha256(train_score),"temperature":float(temperature),"additive_gauge_centered":True,"exact_hilbert_tropical_projective_metric":True,"exact_equal_group_class_medoids":True,"direct_distance_contrast_no_fitted_head":True,"strict_train_iqr_temperature":True,"model_fit_uses_target_rows":False,"target_labels_used":False,"label_independent_exact_transform":True,"local_differential_geometry_not_global_path_integral":False,"diffusion_map_or_nystrom":False,"gmrf_precision_or_logdet":False,"monogenic_riesz_or_phase":False};return model,audit
def tropical_probability(path,model):
    x=gauge(path);distance=projective_distance(x,model["prototypes"]);score=(distance[:,0]-distance[:,1])/model["temperature"];p=1./(1.+np.exp(-np.clip(score,-SCORE_CLIP,SCORE_CLIP)));require(np.isfinite(p).all(),"V218 probability");return np.clip(p,1e-6,1-1e-6),score
def tropical_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);ts,ta=processor.transform(ordered);xs,xa=processor.transform(target);tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]);xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]);gp,y,grouping=equal_event_group_path(ordered,tp);scale=TrainChannelScale().fit(gp);gp=scale.transform(gp);xp=scale.transform(xp);model,geometry=fit_tropical(gp,y);train_p,train_score=tropical_probability(gp,model);p,score=tropical_probability(xp,model);geometry["train_probability_sha256"]=array_sha256(train_p);target_geometry={**geometry,"rows":len(xp),"score_sha256":array_sha256(score),"model_fit_uses_target_rows":False,"target_medoid_or_temperature_fit":False}
    return p,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":geometry,"target_geometry":target_geometry,"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"type":"identity after train-only channel scale and exact per-row additive gauge"},"head":{"type":"none; direct tropical class-medoid distance contrast","temperature":model["temperature"]},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_rows_used_for_channel_scale_medoid_or_temperature":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"diffusion_gmrf_riesz_mca_rbm_dpp_or_ecf":False,"gpu_model_calls":0,"prediction":{"mean":float(p.mean()),"std":float(p.std()),"probability_sha256":array_sha256(p)}}

_BASE_BOOTSTRAP=base.paired_nested_bootstrap
def configure_base():base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=FEATURE_DIMENSION;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=tropical_direction;base.choose_inner=choose_inner
def choose_inner(tr,va,ch):
    candidate,a=tropical_direction(tr,va);b=ch.prob.to_numpy(float);conf=ch.confidence_signal.to_numpy(float);high=bool_series(ch.high_conf).to_numpy(bool);bm=metric(va,b,conf,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}];g=a["train_geometry"];ok=bool(g["additive_gauge_centered"]and g["exact_hilbert_tropical_projective_metric"]and g["exact_equal_group_class_medoids"]and g["direct_distance_contrast_no_fitted_head"]and g["strict_train_iqr_temperature"]and not g["model_fit_uses_target_rows"]and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(b,candidate,ar["weight"]);m=metric(va,p,conf,high);auc=m["auc"]-bm["auc"];ba=m["balanced_accuracy"]-bm["balanced_accuracy"];net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"];floor,d=base.supported_source_ba_delta(va,b,p);trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    s=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return{"selection_rule":"inner OOF V69 noop or fixed .25/.50 tropical-projective class-medoid blend","baseline":bm,"selected":s,"trials":trials,"outer_labels_used_for_selection":False,"metric_medoid_temperature_blend_or_safety_micro_tuning":False},a
def run_nested(dev,ch,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):d,e,a=base.run_nested(dev,ch,smoke,smoke_market)
    d=d.rename(columns={"v146_model":"v218_model"});e=e.rename(columns={"frenet_prob":"tropical_projective_medoid_prob","v146_model":"v218_model"})
    for x in a:s=x["policy"]["selected"];print(f"[V218 TROPICAL MEDOID] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws):configure_base();r=_BASE_BOOTSTRAP(e,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V218 tropical-projective medoid policy";r["seed"]=SEED;r["standardized"]=True;return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(ch,d,e,a,draws,smoke);c=r["material_checks"];c.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(x[side]["train_geometry"]["additive_gauge_centered"]and x[side]["train_geometry"]["exact_hilbert_tropical_projective_metric"]and x[side]["train_geometry"]["exact_equal_group_class_medoids"]and x[side]["train_geometry"]["direct_distance_contrast_no_fitted_head"]and x[side]["train_geometry"]["strict_train_iqr_temperature"]and not x[side]["model_fit_uses_target_rows"]and not x[side]["target_labels_used"]for x in a for side in("inner_model_audit","outer_model_audit"));c["multichannel_causal_tropical_projective_class_medoid_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]:r["selected_frame"]=ch.copy();r["selected_summary"]=r["baseline_summary"];require(r["selected_frame"].equals(ch),"V218 fallback")
    return r
def material_gate(e):c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v)for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key");return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V218").resolve(),"binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate;reports={"MODEL_COMPARISON.json":{"version":"V218","models":{"V69":e["baseline_summary"],"V218_DIAGNOSTIC":e["candidate_summary"],"V218_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V218","status":e["status"],"selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V218","selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V218_TROPICAL_MEDOID_REPORT.json":{"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V218_TROPICAL_MEDOID_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V218_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"controller noarg conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full")
    return a
def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);require(bounded or CONTROLLER_RESOURCE_MODE,"V218 unbound full resource mode");runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller inherited full resources tropical medoid","gpu_model_calls":0,"numeric_thread_cap":None,"resource_mutation":False}
    if bounded:
        if os.name=="nt":require(_kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)),"V218 bounded runtime affinity")
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    runtime["observed_affinity"]=observed_affinity(bounded);runtime["gpu_visible"]=os.environ.get("CUDA_VISIBLE_DEVICES");runtime["gpu_model_calls"]=0;authority=verify_authority();dev,ch=load_authorized()
    if a.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"nested_key":"selected.material_gate.nested_bootstrap","native_robustness":True,"exact_fallback":True},"output_written":False}),indent=2));return
    pool_context=threadpool_limits(limits=2)if bounded else nullcontext()
    with pool_context:
        d,e,aud=run_nested(dev,ch,bounded,a.smoke_market)
        if a.support_probe:
            x=aud[0];non=[v for v in x["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":x["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":x["outer_baseline"],"raw_outer_selected":x["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":x["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":x["inner_chronology"],"strict_outer_chronology":x["outer_chronology"],"output_written":False}),indent=2));return
        ev=evaluate(ch,d,e,aud,SMOKE_BOOTSTRAP_DRAWS if a.smoke_test else BOOTSTRAP_DRAWS,a.smoke_test)
    non=[v for v in aud[0]["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));s={"status":"SMOKE_OK"if a.smoke_test else ev["status"],"runtime":runtime,"selected_models":[x["policy"]["selected"]["name"]for x in aud],"outer_auc_delta":ev["outer_auc_delta"],"outer_ba_delta":ev["outer_ba_delta"],"outer_net_delta":ev["outer_net_delta"],"material_pass":ev["material_pass"],"fallback_is_exact_v69":not ev["material_pass"],"current_material_gate":material_gate(ev),"output_written":False}
    if a.smoke_test:s.update({"raw_inner_selected":aud[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":aud[0]["outer_baseline"],"raw_outer_selected":aud[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":aud[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(s),indent=2));return
    print(json.dumps(clean({**s,"output_written":True,"controller":write_outputs(a.output,authority,e,aud,ev)}),indent=2))
if __name__=="__main__":main()
