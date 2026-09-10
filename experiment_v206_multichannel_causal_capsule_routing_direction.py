"""V206 deterministic routing-by-agreement capsule direction challenger."""
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
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
import experiment_v198_multichannel_causal_score_matching_energy_direction as scaffold

core=scaffold.scaffold;base=scaffold.base;ROOT=scaffold.ROOT;VERSION=206;HYPOTHESIS="MULTICHANNEL_CAUSAL_CAPSULE_ROUTING_DIRECTION_V1";DEFAULT_OUT=ROOT/"research"/"staging"/"V206"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v198_multichannel_causal_score_matching_energy_direction.py";EXPECTED_SCAFFOLD_SHA256="80338c3f473670813ee1a888958b44911d9e926ba469a455ed48b086514ef9bc"
PRIMARY_N=16;PRIMARY_DIM=4;CLASS_N=2;CLASS_DIM=8;ROUTING_ITERATIONS=3;BOUNDED_EPOCHS=8;FULL_EPOCHS=48;LEARNING_RATE=.01;WEIGHT_DECAY=.0005;GRADIENT_CLIP=5.;SEED=20601;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"CAPSULE_ROUTING_W0.25","weight":.25},{"name":"CAPSULE_ROUTING_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
PARAMETER_N=PRIMARY_DIM*PRIMARY_DIM+PRIMARY_DIM+PRIMARY_N*CLASS_N*CLASS_DIM*PRIMARY_DIM
STATIC_SPEC={"feature_dimension":CLASS_N*CLASS_DIM,"primary_capsules":"sixteen nonoverlapping 2x2 patches through shared learned 4-to-4 tanh map","direction_capsules":"two learned 8D class capsules","routing":"exactly three deterministic routing-by-agreement iterations with squash nonlinearity","objective":"equal-event-group/class-balanced capsule margin loss m+=0.9 m-=0.1 lambda=0.5","head":"direct normalized class-capsule lengths","bounded_epochs":BOUNDED_EPOCHS,"full_epochs":FULL_EPOCHS,"parameter_count":PARAMETER_N,"target_model_fit":False}
require(PARAMETER_N==1044,"V206 parameter count")

class CapsuleDirection(nn.Module):
    def __init__(self):
        super().__init__();self.primary=nn.Linear(PRIMARY_DIM,PRIMARY_DIM);self.vote=nn.Parameter(torch.empty(PRIMARY_N,CLASS_N,CLASS_DIM,PRIMARY_DIM));generator=torch.Generator(device="cpu");generator.manual_seed(SEED)
        with torch.no_grad():self.primary.weight.copy_(.08*torch.randn(self.primary.weight.shape,generator=generator));self.primary.bias.zero_();self.vote.copy_(.08*torch.randn(self.vote.shape,generator=generator))
    @staticmethod
    def squash(value):
        norm2=(value*value).sum(dim=-1,keepdim=True);return norm2/(1.+norm2)*value/torch.sqrt(norm2+1e-8)
    def forward(self,patches):
        primary=torch.tanh(self.primary(patches));votes=torch.einsum("nip,icop->nico",primary,self.vote);logits=torch.zeros(votes.shape[:3],dtype=votes.dtype,device=votes.device)
        for iteration in range(ROUTING_ITERATIONS):
            coefficient=torch.softmax(logits,dim=2);capsule=self.squash((coefficient[:,:,:,None]*votes).sum(dim=1))
            if iteration+1<ROUTING_ITERATIONS:logits=logits+(votes*capsule[:,None,:,:]).sum(dim=-1)
        return torch.linalg.vector_norm(capsule,dim=-1)

def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V198 scaffold changed");a=scaffold.verify_authority();a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V198 output"};a["v206_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return a
def setup_device():
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.use_deterministic_algorithms(True);use_cuda=bool(CONTROLLER_OUTPUT and os.environ.get("CUDA_VISIBLE_DEVICES")=="0")
    if use_cuda:require(torch.cuda.is_available()and torch.cuda.device_count()>=1,"V206 CUDA0 unavailable");torch.cuda.set_device(0);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return torch.device("cuda:0"if use_cuda else"cpu"),FULL_EPOCHS if use_cuda else BOUNDED_EPOCHS
def patches(path):return path.reshape(len(path),4,2,4,2).transpose(0,1,3,2,4).reshape(len(path),PRIMARY_N,PRIMARY_DIM)
def capsule_loss(length,label):
    one=F.one_hot(label,num_classes=CLASS_N).to(length.dtype);term=one*F.relu(.9-length).square()+.5*(1.-one)*F.relu(length-.1).square();class_n=torch.bincount(label,minlength=2).to(length.dtype);weight=torch.where(label==1,.5/class_n[1],.5/class_n[0]);return (weight*term.sum(dim=1)).sum()
def parameter_sha(model):return array_sha256(np.concatenate([p.detach().cpu().numpy().ravel()for p in model.parameters()]))
def fit_capsule(train_patch,target_patch,y):
    device,epochs=setup_device();torch.manual_seed(SEED)
    if device.type=="cuda":torch.cuda.manual_seed_all(SEED)
    model=CapsuleDirection().to(device);tr=torch.from_numpy(train_patch.astype(np.float32)).to(device);ta=torch.from_numpy(target_patch.astype(np.float32)).to(device);label=torch.from_numpy(y.astype(np.int64)).to(device);require(int(torch.bincount(label,minlength=2).min())>20,"V206 class support");optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    with torch.no_grad():initial=float(capsule_loss(model(tr),label).cpu())
    gradient_max=0.
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True);loss=capsule_loss(model(tr),label);require(bool(torch.isfinite(loss).item()),"V206 loss");loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP).detach().cpu());gradient_max=max(gradient_max,gradient);optimizer.step()
    with torch.no_grad():train_length=model(tr);target_length=model(ta);final=float(capsule_loss(train_length,label).cpu());prob=(target_length[:,1]/torch.clamp(target_length.sum(dim=1),min=1e-8)).cpu().numpy().astype(float)
    require(np.isfinite(prob).all()and final<=initial+1e-4,"V206 training contract");audit={"backend":"torch_cuda0"if device.type=="cuda"else"torch_cpu","epochs":epochs,"parameter_n":sum(p.numel()for p in model.parameters()),"parameter_sha256":parameter_sha(model),"initial_margin_loss":initial,"final_margin_loss":final,"gradient_max":gradient_max,"routing_iterations":ROUTING_ITERATIONS,"routing_by_agreement":True,"capsule_squash":True,"class_capsule_length_probability":True,"model_fit_uses_target_rows":False,"target_labels_used":False};return np.clip(prob,1e-6,1-1e-6),audit
def capsule_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);ts,ta=processor.transform(ordered);xs,xa=processor.transform(target);tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]);xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]);gp,y,grouping=equal_event_group_path(ordered,tp);scale=TrainChannelScale().fit(gp);train_patch=patches(scale.transform(gp));target_patch=patches(scale.transform(xp));prob,training=fit_capsule(train_patch,target_patch,y)
    geometry={"rows":len(train_patch),"feature_dimension":CLASS_N*CLASS_DIM,"nonoverlapping_patch_primary_capsules":True,"primary_capsule_n":PRIMARY_N,"class_capsule_n":CLASS_N,"class_capsule_dim":CLASS_DIM,"routing_by_agreement":True,"routing_iterations":ROUTING_ITERATIONS,"capsule_squash":True,"capsule_margin_loss":True,"model_fit_uses_target_rows":False,"target_labels_used":False,"training":training,"label_independent_exact_transform":False,"local_differential_geometry_not_global_path_integral":False,"v82_autoencoder":False,"v150_mlp_mixer":False,"v153_axial_attention":False,"v192_tensor_train":False,"v199_contrastive_prototypes":False,"v204_choquet_capacity":False,"v205_rasch_latent_ability":False}
    return prob,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":geometry,"target_geometry":{**geometry,"rows":len(target_patch),"model_fit_uses_target_rows":False},"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"fixed_patch_partition":True,"scale_sha256":training["parameter_sha256"]},"head":{"type":"direct normalized class capsule lengths"},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(prob.mean()),"std":float(prob.std()),"probability_sha256":array_sha256(prob)}}

_BASE_BOOTSTRAP=scaffold._BASE_BOOTSTRAP
def configure_base():
    core.VERSION=VERSION;core.HYPOTHESIS=HYPOTHESIS;core.STATIC_SPEC=STATIC_SPEC;core.DAG_FEATURE_DIMENSION=CLASS_N*CLASS_DIM;core.ARCHITECTURES=ARCHITECTURES;core.SEED=SEED;core.dag_direction=capsule_direction;core.choose_inner=choose_inner;base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=CLASS_N*CLASS_DIM;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=capsule_direction
def choose_inner(tr,va,ch):
    challenger,a=capsule_direction(tr,va);b=ch.prob.to_numpy(float);conf=ch.confidence_signal.to_numpy(float);high=bool_series(ch.high_conf).to_numpy(bool);bm=metric(va,b,conf,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}];g=a["train_geometry"];ok=bool(g["nonoverlapping_patch_primary_capsules"]and g["routing_by_agreement"]and g["capsule_squash"]and g["capsule_margin_loss"]and not g["model_fit_uses_target_rows"]and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(b,challenger,ar["weight"]);m=metric(va,p,conf,high);auc=m["auc"]-bm["auc"];ba=m["balanced_accuracy"]-bm["balanced_accuracy"];net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"];floor,d=base.supported_source_ba_delta(va,b,p);trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    s=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return{"selection_rule":"inner OOF V69 noop or fixed .25/.50 capsule routing direction blend","baseline":bm,"selected":s,"trials":trials,"outer_labels_used_for_selection":False,"patch_capsule_dimension_routing_iteration_epoch_margin_blend_or_safety_micro_tuning":False},a
def run_nested(dev,ch,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):d,e,a=core.run_nested(dev,ch,smoke,smoke_market)
    d=d.rename(columns={"v195_model":"v206_model"});e=e.rename(columns={"sparse_dag_moment_prob":"capsule_routing_prob","v195_model":"v206_model"})
    for x in a:
        s=x["policy"]["selected"];print(f"[V206 CAPSULE] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws):configure_base();r=_BASE_BOOTSTRAP(e,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V206 capsule routing policy";r["seed"]=SEED;r["standardized"]=True;return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(ch,d,e,a,draws,smoke);c=r["material_checks"];c.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(x[side]["train_geometry"]["routing_by_agreement"]and x[side]["train_geometry"]["capsule_squash"]and x[side]["train_geometry"]["capsule_margin_loss"]and not x[side]["model_fit_uses_target_rows"]and not x[side]["target_labels_used"]for x in a for side in("inner_model_audit","outer_model_audit"));c["multichannel_causal_capsule_routing_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]:r["selected_frame"]=ch.copy();r["selected_summary"]=r["baseline_summary"];require(r["selected_frame"].equals(ch),"V206 fallback")
    return r
def material_gate(e):c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v)for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key");return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V206").resolve(),"binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate
    reports={"MODEL_COMPARISON.json":{"version":"V206","models":{"V69":e["baseline_summary"],"V206_DIAGNOSTIC":e["candidate_summary"],"V206_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V206","selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V206","selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V206_CAPSULE_ROUTING_REPORT.json":{"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V206_CAPSULE_ROUTING_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V206_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full");return a
    return a
def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller deterministic CUDA0 capsule routing","gpu_model_calls":"capsule model training"};authority=verify_authority();dev,ch=load_authorized()
    if a.full_run:require(os.environ.get("CUDA_VISIBLE_DEVICES")=="0"and torch.cuda.is_available(),"V206 full CUDA0 unavailable")
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
