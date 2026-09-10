"""V202 analytic class-conditional kernel Stein discrepancy direction."""
from __future__ import annotations
import os
CONTROLLER_OUTPUT=os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _n in("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):os.environ[_n]="2"
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import experiment_v198_multichannel_causal_score_matching_energy_direction as scaffold

core=scaffold.scaffold;base=scaffold.base;ROOT=scaffold.ROOT;VERSION=202;HYPOTHESIS="MULTICHANNEL_CAUSAL_KERNEL_STEIN_DISCREPANCY_DIRECTION_V1";DEFAULT_OUT=ROOT/"research"/"staging"/"V202"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v198_multichannel_causal_score_matching_energy_direction.py";EXPECTED_SCAFFOLD_SHA256="80338c3f473670813ee1a888958b44911d9e926ba469a455ed48b086514ef9bc"
PROJECTION_DIM=16;BANDWIDTH_FLOOR=.25;BATCH_SIZE=256;SEED=20201;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"KERNEL_STEIN_DISCREPANCY_W0.25","weight":.25},{"name":"KERNEL_STEIN_DISCREPANCY_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
STATIC_SPEC={"feature_dimension":PROJECTION_DIM,"projection":"fixed label-independent nonoverlapping 2x2 average pooling of 8x8 surface","class_score":"analytic isotropic Gaussian-mixture score from equal-event-group class references","bandwidth":"single pooled strict-train median pairwise-distance heuristic with floor 0.25","discrepancy":"singleton RBF kernel Stein discrepancy sqrt(||score||^2+d/h^2)","head":"direct log DOWN-KSD minus log UP-KSD with strict-train IQR temperature","target_model_fit":False}

def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V198 scaffold changed");a=scaffold.verify_authority();a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V198 output"};a["v202_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return a
def setup_device():
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    use_cuda=bool(CONTROLLER_OUTPUT and os.environ.get("CUDA_VISIBLE_DEVICES")=="0")
    if use_cuda:require(torch.cuda.is_available()and torch.cuda.device_count()>=1,"V202 CUDA0 unavailable")
    return torch.device("cuda:0"if use_cuda else"cpu")
def compress_surface(x):return x.reshape(len(x),4,2,4,2).mean(axis=(2,4)).reshape(len(x),PROJECTION_DIM)
def pooled_bandwidth(x):
    d2=((x[:,None,:]-x[None,:,:])**2).sum(axis=2);tri=d2[np.triu_indices(len(x),1)];positive=tri[tri>1e-12];require(len(positive)>20,"V202 bandwidth support");return max(float(np.sqrt(.5*np.median(positive))),BANDWIDTH_FLOOR)
def class_ksd(query,reference,bandwidth,device):
    ref=torch.from_numpy(reference.astype(np.float32)).to(device);h2=float(bandwidth**2);out=[]
    for start in range(0,len(query),BATCH_SIZE):
        q=torch.from_numpy(query[start:start+BATCH_SIZE].astype(np.float32)).to(device);diff=q[:,None,:]-ref[None,:,:];log_kernel=-.5*(diff*diff).sum(dim=2)/h2;weight=torch.softmax(log_kernel,dim=1);score=-(weight[:,:,None]*diff).sum(dim=1)/h2;ksd=torch.sqrt(torch.clamp((score*score).sum(dim=1)+PROJECTION_DIM/h2,min=1e-12));out.append(ksd.detach().cpu().numpy())
    return np.concatenate(out).astype(float)
def kernel_stein_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);ts,ta=processor.transform(ordered);xs,xa=processor.transform(target);tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]);xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]);gp,y,grouping=equal_event_group_path(ordered,tp);scale=TrainChannelScale().fit(gp);train_x=compress_surface(scale.transform(gp));target_x=compress_surface(scale.transform(xp));bandwidth=pooled_bandwidth(train_x);device=setup_device();references=[train_x[y==cls]for cls in(0,1)];require(min(map(len,references))>20,"V202 class support");train_ksd=[class_ksd(train_x,r,bandwidth,device)for r in references];target_ksd=[class_ksd(target_x,r,bandwidth,device)for r in references];train_margin=np.log(train_ksd[0]+1e-12)-np.log(train_ksd[1]+1e-12);target_margin=np.log(target_ksd[0]+1e-12)-np.log(target_ksd[1]+1e-12);q25,q75=np.quantile(train_margin,[.25,.75]);temperature=max(float((q75-q25)/1.349),.05);prob=np.clip(1/(1+np.exp(-np.clip(target_margin/temperature,-30,30))),1e-6,1-1e-6)
    geometry={"rows":len(train_x),"feature_dimension":PROJECTION_DIM,"backend":"torch_cuda0"if device.type=="cuda"else"torch_cpu","class_conditional_analytic_gaussian_mixture_scores":True,"kernel_stein_discrepancy":True,"singleton_rbf_ksd_formula":True,"pooled_train_only_bandwidth":True,"bandwidth":bandwidth,"bandwidth_floor":BANDWIDTH_FLOOR,"reference_n":{"DOWN":len(references[0]),"UP":len(references[1])},"reference_sha256":{"DOWN":array_sha256(references[0]),"UP":array_sha256(references[1])},"model_fit_uses_target_rows":False,"target_labels_used":False,"label_independent_exact_transform":False,"local_differential_geometry_not_global_path_integral":False,"v84_random_fourier_kernel":False,"v154_nystrom_rbf_pairwise_auc":False,"v198_learned_energy_score":False,"v201_continuous_normalizing_flow":False,"mmd_mean_embedding_classifier":False}
    return prob,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":geometry,"target_geometry":{**geometry,"rows":len(target_x),"model_fit_uses_target_rows":False},"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"fixed_average_pool":True,"bandwidth":bandwidth,"scale_sha256":array_sha256(np.array([temperature]))},"head":{"type":"direct class KSD contrast","temperature":temperature},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(prob.mean()),"std":float(prob.std()),"probability_sha256":array_sha256(prob)}}

_BASE_BOOTSTRAP=scaffold._BASE_BOOTSTRAP
def configure_base():
    core.VERSION=VERSION;core.HYPOTHESIS=HYPOTHESIS;core.STATIC_SPEC=STATIC_SPEC;core.DAG_FEATURE_DIMENSION=PROJECTION_DIM;core.ARCHITECTURES=ARCHITECTURES;core.SEED=SEED;core.dag_direction=kernel_stein_direction;core.choose_inner=choose_inner;base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=PROJECTION_DIM;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=kernel_stein_direction
def choose_inner(tr,va,ch):
    challenger,a=kernel_stein_direction(tr,va);b=ch.prob.to_numpy(float);conf=ch.confidence_signal.to_numpy(float);high=bool_series(ch.high_conf).to_numpy(bool);bm=metric(va,b,conf,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}];g=a["train_geometry"];ok=bool(g["class_conditional_analytic_gaussian_mixture_scores"]and g["kernel_stein_discrepancy"]and g["singleton_rbf_ksd_formula"]and g["pooled_train_only_bandwidth"]and not g["model_fit_uses_target_rows"]and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(b,challenger,ar["weight"]);m=metric(va,p,conf,high);auc=m["auc"]-bm["auc"];ba=m["balanced_accuracy"]-bm["balanced_accuracy"];net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"];floor,d=base.supported_source_ba_delta(va,b,p);trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    s=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return{"selection_rule":"inner OOF V69 noop or fixed .25/.50 analytic class kernel Stein discrepancy blend","baseline":bm,"selected":s,"trials":trials,"outer_labels_used_for_selection":False,"projection_kernel_bandwidth_reference_discrepancy_blend_or_safety_micro_tuning":False},a
def run_nested(dev,ch,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):d,e,a=core.run_nested(dev,ch,smoke,smoke_market)
    d=d.rename(columns={"v195_model":"v202_model"});e=e.rename(columns={"sparse_dag_moment_prob":"kernel_stein_discrepancy_prob","v195_model":"v202_model"})
    for x in a:
        s=x["policy"]["selected"];print(f"[V202 KERNEL STEIN] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws):configure_base();r=_BASE_BOOTSTRAP(e,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V202 kernel Stein policy";r["seed"]=SEED;r["standardized"]=True;return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(ch,d,e,a,draws,smoke);c=r["material_checks"];c.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(x[side]["train_geometry"]["class_conditional_analytic_gaussian_mixture_scores"]and x[side]["train_geometry"]["kernel_stein_discrepancy"]and x[side]["train_geometry"]["pooled_train_only_bandwidth"]and not x[side]["model_fit_uses_target_rows"]and not x[side]["target_labels_used"]for x in a for side in("inner_model_audit","outer_model_audit"));c["multichannel_causal_kernel_stein_discrepancy_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]:r["selected_frame"]=ch.copy();r["selected_summary"]=r["baseline_summary"];require(r["selected_frame"].equals(ch),"V202 fallback")
    return r
def material_gate(e):c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v)for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key");return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V202").resolve(),"binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate
    reports={"MODEL_COMPARISON.json":{"version":"V202","models":{"V69":e["baseline_summary"],"V202_DIAGNOSTIC":e["candidate_summary"],"V202_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V202","selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V202","selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V202_KERNEL_STEIN_DISCREPANCY_REPORT.json":{"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V202_KERNEL_STEIN_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V202_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full");return a
    return a
def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller deterministic CUDA0 analytic KSD","gpu_model_calls":"class KDE score and KSD tensors"};authority=verify_authority();dev,ch=load_authorized()
    if a.full_run:require(os.environ.get("CUDA_VISIBLE_DEVICES")=="0"and torch.cuda.is_available(),"V202 full CUDA0 unavailable")
    if a.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"full_cuda0_fail_closed":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"nested_key":"selected.material_gate.nested_bootstrap","exact_fallback":True},"output_written":False}),indent=2));return
    with threadpool_limits(limits=2 if bounded else None):
        d,e,aud=run_nested(dev,ch,bounded,a.smoke_market)
        if a.support_probe:
            x=aud[0];non=[v for v in x["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":x["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":x["outer_baseline"],"raw_outer_selected":x["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":x["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":x["inner_chronology"],"strict_outer_chronology":x["outer_chronology"],"output_written":False}),indent=2));return
        ev=evaluate(ch,d,e,aud,250 if a.smoke_test else 2000,a.smoke_test)
    non=[v for v in aud[0]["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));s={"status":"SMOKE_OK"if a.smoke_test else ev["status"],"runtime":runtime,"selected_models":[x["policy"]["selected"]["name"]for x in aud],"outer_auc_delta":ev["outer_auc_delta"],"outer_ba_delta":ev["outer_ba_delta"],"outer_net_delta":ev["outer_net_delta"],"material_pass":ev["material_pass"],"fallback_is_exact_v69":not ev["material_pass"],"current_material_gate":material_gate(ev),"output_written":False}
    if a.smoke_test:s.update({"raw_inner_selected":aud[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":aud[0]["outer_baseline"],"raw_outer_selected":aud[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":aud[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(s),indent=2));return
    print(json.dumps(clean({**s,"output_written":True,"controller":write_outputs(a.output,authority,e,aud,ev)}),indent=2))
if __name__=="__main__":main()
