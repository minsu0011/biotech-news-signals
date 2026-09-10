"""V212 strict-train multiple correspondence analysis direction."""
from __future__ import annotations
import os
import ctypes
CPU_AFFINITY_MASK=0xC0000000
if os.name=="nt":
    _kernel32=ctypes.windll.kernel32;_kernel32.GetCurrentProcess.restype=ctypes.c_void_p;_kernel32.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t);_kernel32.SetProcessAffinityMask.restype=ctypes.c_bool;_kernel32.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t));_kernel32.GetProcessAffinityMask.restype=ctypes.c_bool;_process=_kernel32.GetCurrentProcess()
    if not _kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)):raise RuntimeError("V212 pre-import affinity")
for _name in("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):os.environ[_name]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import experiment_v209_multichannel_causal_cross_horizon_regularized_cca_direction as scaffold
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
if os.name=="nt" and not _kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)):raise RuntimeError("V212 post-scaffold affinity")

CONTROLLER_OUTPUT=os.environ.get("MARKET_BIO_VERSION_OUTPUT");base=scaffold.base;ROOT=scaffold.ROOT;VERSION=212;HYPOTHESIS="MULTICHANNEL_CAUSAL_MULTIPLE_CORRESPONDENCE_ANALYSIS_DIRECTION_V1";DEFAULT_OUT=ROOT/"research"/"staging"/"V212"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v209_multichannel_causal_cross_horizon_regularized_cca_direction.py";EXPECTED_SCAFFOLD_SHA256="60d45f438d985ac67688b19e970a741b2098a26c01e58b35585004931eba68ab"
SITE_N=64;STATE_N=3;INDICATOR_DIMENSION=192;MCA_RANK=16;FEATURE_DIMENSION=16;RIDGE_LOGISTIC_C=.5;FEATURE_CLIP=8.;SEED=21201;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"MULTIPLE_CORRESPONDENCE_W0.25","weight":.25},{"name":"MULTIPLE_CORRESPONDENCE_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
STATIC_SPEC={"path_shape":[8,8],"states":"strict-train pooled per-site tertiles","indicator_dimension":INDICATOR_DIMENSION,"geometry":"complete disjunctive table, fixed Jeffreys-smoothed category-mass chi-square centering and normalization","mca_rank":MCA_RANK,"feature_dimension":FEATURE_DIMENSION,"head":"equal-event-group/class-balanced C=.5 ridge logistic","target_model_fit":False,"gpu_model_calls":0}

def observed_affinity():
    if os.name!="nt":return{"mask_hex":None,"cpus":None}
    pm=ctypes.c_size_t();sm=ctypes.c_size_t();require(_kernel32.GetProcessAffinityMask(_process,ctypes.byref(pm),ctypes.byref(sm)),"V212 affinity query");cpus=[i for i in range(64)if pm.value&(1<<i)];require(pm.value==CPU_AFFINITY_MASK and cpus==[30,31],"V212 affinity drift");return{"mask_hex":f"0x{pm.value:08X}","cpus":cpus}
def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V209 scaffold changed");a=scaffold.verify_authority();a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V209 output"};a["v212_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return a

def states_from_thresholds(path,thresholds):
    flat=path.reshape(len(path),SITE_N);return(flat>thresholds[0]).astype(np.int8)+(flat>thresholds[1]).astype(np.int8)
def indicator(states):
    result=np.eye(STATE_N,dtype=float)[states].reshape(len(states),INDICATOR_DIMENSION);require(np.allclose(result.reshape(len(states),SITE_N,STATE_N).sum(2),1.),"V212 disjunctive row");return result
def fit_mca(path):
    flat=path.reshape(len(path),SITE_N);thresholds=np.quantile(flat,[1/3,2/3],axis=0);states=states_from_thresholds(path,thresholds);z=indicator(states);mass=(z.sum(0)+.5)/(len(z)+1.5);require(float(mass.min())>0.,"V212 empty category");standardized=(z-mass)/np.sqrt(mass);_,singular,right_t=np.linalg.svd(standardized/np.sqrt(len(z)),full_matrices=False);components=right_t[:MCA_RANK].copy()
    for k in range(MCA_RANK):
        anchor=int(np.argmax(np.abs(components[k])));
        if components[k,anchor]<0:components[k]*=-1
    model={"thresholds":thresholds,"mass":mass,"components":components};audit={"rows":len(path),"feature_dimension":FEATURE_DIMENSION,"indicator_dimension":INDICATOR_DIMENSION,"mca_rank":MCA_RANK,"threshold_sha256":array_sha256(thresholds),"category_mass_sha256":array_sha256(mass),"component_sha256":array_sha256(components),"principal_inertias":(singular[:MCA_RANK]**2).tolist(),"minimum_category_mass":float(mass.min()),"complete_disjunctive_table":True,"jeffreys_smoothed_category_mass":True,"category_mass_chi_square_geometry":True,"label_free_train_only_mca_svd":True,"target_threshold_mass_or_component_fit":False,"model_fit_uses_target_rows":False,"target_labels_used":False,"label_independent_exact_transform":True,"local_differential_geometry_not_global_path_integral":False,"bernoulli_rbm_or_free_energy":False,"within_event_hsic_or_kernel_alignment":False,"cross_horizon_cca":False};return model,audit
def mca_features(path,model):
    z=indicator(states_from_thresholds(path,model["thresholds"]));feature=((z-model["mass"])/np.sqrt(model["mass"]))@model["components"].T;require(feature.shape==(len(path),FEATURE_DIMENSION)and np.isfinite(feature).all(),"V212 MCA feature");return feature
class RobustFeatureScale:
    def fit(self,x):self.median=np.median(x,0);q25,q75=np.quantile(x,[.25,.75],axis=0);r=(q75-q25)/1.349;s=np.std(x,0);self.scale=np.where(r>1e-8,r,np.where(s>1e-8,s,1.));return self
    def transform(self,x):z=np.clip((x-self.median)/self.scale,-FEATURE_CLIP,FEATURE_CLIP);require(np.isfinite(z).all(),"V212 feature scale");return z
def mca_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);ts,ta=processor.transform(ordered);xs,xa=processor.transform(target);tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]);xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]);gp,y,grouping=equal_event_group_path(ordered,tp);scale=TrainChannelScale().fit(gp);gp=scale.transform(gp);xp=scale.transform(xp);model,geometry=fit_mca(gp);tf=mca_features(gp,model);xf=mca_features(xp,model);fs=RobustFeatureScale().fit(tf);td=fs.transform(tf);xd=fs.transform(xf);n=np.bincount(y,minlength=2).astype(float);require(n.min()>0,"V212 class");weight=np.where(y==1,.5/n[1],.5/n[0]);head=LogisticRegression(C=RIDGE_LOGISTIC_C,penalty="l2",solver="liblinear",max_iter=1000,random_state=SEED);head.fit(td,y,sample_weight=weight);p=np.clip(head.predict_proba(xd)[:,1],1e-6,1-1e-6);geometry["feature_sha256"]=array_sha256(tf);target_geometry={**geometry,"rows":len(xf),"feature_sha256":array_sha256(xf),"model_fit_uses_target_rows":False,"target_threshold_mass_or_component_fit":False}
    return p,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":geometry,"target_geometry":target_geometry,"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"median_sha256":array_sha256(fs.median),"scale_sha256":array_sha256(fs.scale),"clip":FEATURE_CLIP},"head":{"type":"equal-group class-balanced ridge logistic","C":RIDGE_LOGISTIC_C,"coefficient_sha256":array_sha256(head.coef_),"iterations":int(head.n_iter_[0])},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"gpu_model_calls":0,"prediction":{"mean":float(p.mean()),"std":float(p.std()),"probability_sha256":array_sha256(p)}}

_BASE_BOOTSTRAP=base.paired_nested_bootstrap
def configure_base():base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=FEATURE_DIMENSION;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=mca_direction;base.choose_inner=choose_inner
def choose_inner(tr,va,ch):
    candidate,a=mca_direction(tr,va);b=ch.prob.to_numpy(float);conf=ch.confidence_signal.to_numpy(float);high=bool_series(ch.high_conf).to_numpy(bool);bm=metric(va,b,conf,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}];g=a["train_geometry"];ok=bool(g["complete_disjunctive_table"]and g["category_mass_chi_square_geometry"]and g["label_free_train_only_mca_svd"]and not g["model_fit_uses_target_rows"]and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(b,candidate,ar["weight"]);m=metric(va,p,conf,high);auc=m["auc"]-bm["auc"];ba=m["balanced_accuracy"]-bm["balanced_accuracy"];net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"];floor,d=base.supported_source_ba_delta(va,b,p);trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    s=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return{"selection_rule":"inner OOF V69 noop or fixed .25/.50 MCA blend","baseline":bm,"selected":s,"trials":trials,"outer_labels_used_for_selection":False,"tertile_rank_geometry_head_blend_or_safety_micro_tuning":False},a
def run_nested(dev,ch,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):d,e,a=base.run_nested(dev,ch,smoke,smoke_market)
    d=d.rename(columns={"v146_model":"v212_model"});e=e.rename(columns={"frenet_prob":"multiple_correspondence_prob","v146_model":"v212_model"})
    for x in a:s=x["policy"]["selected"];print(f"[V212 MCA] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws):configure_base();r=_BASE_BOOTSTRAP(e,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V212 MCA policy";r["seed"]=SEED;r["standardized"]=True;return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(ch,d,e,a,draws,smoke);c=r["material_checks"];c.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(x[side]["train_geometry"]["complete_disjunctive_table"]and x[side]["train_geometry"]["category_mass_chi_square_geometry"]and x[side]["train_geometry"]["label_free_train_only_mca_svd"]and not x[side]["model_fit_uses_target_rows"]and not x[side]["target_labels_used"]for x in a for side in("inner_model_audit","outer_model_audit"));c["multichannel_causal_multiple_correspondence_analysis_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]:r["selected_frame"]=ch.copy();r["selected_summary"]=r["baseline_summary"];require(r["selected_frame"].equals(ch),"V212 fallback")
    return r
def material_gate(e):c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v)for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key");return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V212").resolve(),"binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate;reports={"MODEL_COMPARISON.json":{"version":"V212","models":{"V69":e["baseline_summary"],"V212_DIAGNOSTIC":e["candidate_summary"],"V212_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V212","status":e["status"],"selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V212","selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V212_MCA_REPORT.json":{"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V212_MCA_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V212_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"controller noarg conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full")
    return a
def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller deterministic CPU MCA","gpu_model_calls":0}
    if os.name=="nt":require(_kernel32.SetProcessAffinityMask(_process,ctypes.c_size_t(CPU_AFFINITY_MASK)),"V212 runtime affinity")
    os.environ["CUDA_VISIBLE_DEVICES"]="-1";runtime["observed_affinity"]=observed_affinity();runtime["gpu_visible"]="-1";runtime["gpu_model_calls"]=0;authority=verify_authority();dev,ch=load_authorized()
    if a.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"nested_key":"selected.material_gate.nested_bootstrap","native_robustness":True,"exact_fallback":True},"output_written":False}),indent=2));return
    with threadpool_limits(limits=2):
        d,e,aud=run_nested(dev,ch,bounded,a.smoke_market)
        if a.support_probe:
            x=aud[0];non=[v for v in x["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":x["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":x["outer_baseline"],"raw_outer_selected":x["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":x["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":x["inner_chronology"],"strict_outer_chronology":x["outer_chronology"],"output_written":False}),indent=2));return
        ev=evaluate(ch,d,e,aud,SMOKE_BOOTSTRAP_DRAWS if a.smoke_test else BOOTSTRAP_DRAWS,a.smoke_test)
    non=[v for v in aud[0]["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));s={"status":"SMOKE_OK"if a.smoke_test else ev["status"],"runtime":runtime,"selected_models":[x["policy"]["selected"]["name"]for x in aud],"outer_auc_delta":ev["outer_auc_delta"],"outer_ba_delta":ev["outer_ba_delta"],"outer_net_delta":ev["outer_net_delta"],"material_pass":ev["material_pass"],"fallback_is_exact_v69":not ev["material_pass"],"current_material_gate":material_gate(ev),"output_written":False}
    if a.smoke_test:s.update({"raw_inner_selected":aud[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":aud[0]["outer_baseline"],"raw_outer_selected":aud[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":aud[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(s),indent=2));return
    print(json.dumps(clean({**s,"output_written":True,"controller":write_outputs(a.output,authority,e,aud,ev)}),indent=2))
if __name__=="__main__":main()
