"""V198 class-conditional denoising score-matching energy direction."""
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
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import experiment_v195_multichannel_causal_causal_discovery_dag_moment_direction as scaffold

base=scaffold.base;ROOT=scaffold.ROOT;VERSION=198;HYPOTHESIS="MULTICHANNEL_CAUSAL_SCORE_MATCHING_ENERGY_DIRECTION_V1";DEFAULT_OUT=ROOT/"research"/"staging"/"V198"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v195_multichannel_causal_causal_discovery_dag_moment_direction.py";EXPECTED_SCAFFOLD_SHA256="7dca457e41501359e80700a2a14514eb64d858b9d4679ce47ecbb1e1ed7d068f"
INPUT_DIM=64;HIDDEN_DIM=24;SIGMA=.15;BOUNDED_EPOCHS=4;FULL_EPOCHS=24;BATCH_SIZE=256;LEARNING_RATE=.001;WEIGHT_DECAY=.0001;SEED=19801;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"SCORE_MATCHING_ENERGY_W0.25","weight":.25},{"name":"SCORE_MATCHING_ENERGY_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
STATIC_SPEC={"input_dimension":64,"feature_dimension":64,"models":"separate scalar EnergyMLP 64-24-tanh-1 for DOWN and UP","objective":"denoising score matching: negative input-gradient of energy versus analytic Gaussian corruption score","sigma":SIGMA,"bounded_epochs":BOUNDED_EPOCHS,"full_epochs":FULL_EPOCHS,"gauge":"subtract class-train mean energy; target never anchors energy","probability":"sigmoid centered DOWN-minus-UP energy divided by train-only robust IQR temperature","head":"direct energy difference, no discriminative direction head","target_model_fit":False}

class EnergyMLP(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(INPUT_DIM,HIDDEN_DIM),nn.Tanh(),nn.Linear(HIDDEN_DIM,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V195 scaffold changed");a=scaffold.verify_authority();a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V195 output"};a["v198_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return a
def setup_device():
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    use_cuda=bool(CONTROLLER_OUTPUT and os.environ.get("CUDA_VISIBLE_DEVICES")=="0")
    if use_cuda:require(torch.cuda.is_available()and torch.cuda.device_count()>=1,"V198 CUDA0 unavailable");torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return torch.device("cuda:0"if use_cuda else"cpu"),FULL_EPOCHS if use_cuda else BOUNDED_EPOCHS
def init_model(device,seed):
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    model=EnergyMLP().to(device)
    for module in model.modules():
        if isinstance(module,nn.Linear):nn.init.xavier_uniform_(module.weight,gain=.7);nn.init.zeros_(module.bias)
    return model
def fit_energy(x:np.ndarray,device,epochs,seed):
    model=init_model(device,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY);clean=torch.from_numpy(x.astype(np.float32));losses=[]
    for epoch in range(epochs):
        generator=torch.Generator(device="cpu");generator.manual_seed(seed+epoch)
        for start in range(0,len(clean),BATCH_SIZE):
            batch=clean[start:start+BATCH_SIZE].to(device);noise=torch.randn(batch.shape,generator=generator,dtype=batch.dtype).to(device);noisy=(batch+SIGMA*noise).detach().requires_grad_(True);energy=model(noisy);score=-torch.autograd.grad(energy.sum(),noisy,create_graph=True)[0];target=-(noisy-batch)/(SIGMA**2);loss=torch.mean((score-target)**2);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach().cpu()))
    require(np.isfinite(losses).all(),"V198 DSM loss");return model,{"epochs":epochs,"initial_loss":losses[0],"final_loss":losses[-1],"steps":len(losses)}
def energies(model,x,device):
    with torch.inference_mode():return model(torch.from_numpy(x.astype(np.float32)).to(device)).cpu().numpy().astype(float)
def energy_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);ts,ta=processor.transform(ordered);xs,xa=processor.transform(target);tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]);xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]);gp,y,grouping=equal_event_group_path(ordered,tp);scale=TrainChannelScale().fit(gp);gp=scale.transform(gp);xp=scale.transform(xp);train_x=gp.reshape(len(gp),-1);target_x=xp.reshape(len(xp),-1);device,epochs=setup_device();models=[];fit_audits=[];offsets=[]
    for cls in(0,1):
        class_x=train_x[y==cls];require(len(class_x)>20,"V198 class support");model,audit=fit_energy(class_x,device,epochs,SEED+100*cls);offset=float(energies(model,class_x,device).mean());models.append(model);fit_audits.append(audit);offsets.append(offset)
    train_margin=(energies(models[0],train_x,device)-offsets[0])-(energies(models[1],train_x,device)-offsets[1]);target_margin=(energies(models[0],target_x,device)-offsets[0])-(energies(models[1],target_x,device)-offsets[1]);q25,q75=np.quantile(train_margin,[.25,.75]);temperature=max(float((q75-q25)/1.349),.05);prob=np.clip(1/(1+np.exp(-np.clip(target_margin/temperature,-30,30))),1e-6,1-1e-6);parameter=np.concatenate([p.detach().cpu().numpy().ravel()for m in models for p in m.parameters()])
    geometry={"rows":len(train_x),"feature_dimension":INPUT_DIM,"backend":"torch_cuda0"if device.type=="cuda"else"torch_cpu","class_conditional_scalar_energy_models":True,"energy_is_input_integrable_by_construction":True,"denoising_score_matching":True,"analytic_gaussian_noise_score_target":True,"sigma":SIGMA,"model_fit_uses_target_rows":False,"target_energy_gauge_fit":False,"class_train_mean_gauge":True,"fit_audits":fit_audits,"parameter_sha256":array_sha256(parameter),"parameter_n":len(parameter),"label_independent_exact_transform":False,"local_differential_geometry_not_global_path_integral":True,"v82_masked_reconstruction_autoencoder":False,"v153_discriminative_attention":False,"v196_realnvp_density_flow":False,"diffusion_time_schedule_or_reverse_sampler":False}
    return prob,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":geometry,"target_geometry":{**geometry,"rows":len(target_x),"model_fit_uses_target_rows":False},"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"median_sha256":array_sha256(np.array(offsets)),"scale_sha256":array_sha256(np.array([temperature])),"clip":30.},"head":{"type":"direct class centered energy difference","temperature":temperature,"class_energy_offsets":offsets},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(prob.mean()),"std":float(prob.std()),"probability_sha256":array_sha256(prob)}}

_BASE_BOOTSTRAP=base.paired_nested_bootstrap
def configure_base():
    scaffold.VERSION=VERSION;scaffold.HYPOTHESIS=HYPOTHESIS;scaffold.STATIC_SPEC=STATIC_SPEC;scaffold.DAG_FEATURE_DIMENSION=INPUT_DIM;scaffold.ARCHITECTURES=ARCHITECTURES;scaffold.SEED=SEED;scaffold.dag_direction=energy_direction;scaffold.choose_inner=choose_inner;base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=INPUT_DIM;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=energy_direction
def choose_inner(tr,va,ch):
    challenger,a=energy_direction(tr,va);b=ch.prob.to_numpy(float);conf=ch.confidence_signal.to_numpy(float);high=bool_series(ch.high_conf).to_numpy(bool);bm=metric(va,b,conf,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}];g=a["train_geometry"];ok=bool(g["class_conditional_scalar_energy_models"]and g["denoising_score_matching"]and g["energy_is_input_integrable_by_construction"]and not g["model_fit_uses_target_rows"]and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(b,challenger,ar["weight"]);m=metric(va,p,conf,high);auc=m["auc"]-bm["auc"];ba=m["balanced_accuracy"]-bm["balanced_accuracy"];net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"];floor,d=base.supported_source_ba_delta(va,b,p);trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    s=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return{"selection_rule":"inner OOF V69 noop or fixed .25/.50 DSM energy blend","baseline":bm,"selected":s,"trials":trials,"outer_labels_used_for_selection":False,"sigma_width_epoch_temperature_blend_or_safety_micro_tuning":False},a
def run_nested(dev,ch,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):d,e,a=scaffold.run_nested(dev,ch,smoke,smoke_market)
    d=d.rename(columns={"v195_model":"v198_model"});e=e.rename(columns={"sparse_dag_moment_prob":"score_matching_energy_prob","v195_model":"v198_model"})
    for x in a:
        s=x["policy"]["selected"];print(f"[V198 DSM ENERGY] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws):configure_base();r=_BASE_BOOTSTRAP(e,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V198 DSM energy policy";r["seed"]=SEED;r["standardized"]=True;return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(ch,d,e,a,draws,smoke);c=r["material_checks"];c.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(x[side]["train_geometry"]["denoising_score_matching"]and x[side]["train_geometry"]["energy_is_input_integrable_by_construction"]and not x[side]["model_fit_uses_target_rows"]and not x[side]["target_labels_used"]for x in a for side in("inner_model_audit","outer_model_audit"));c["multichannel_causal_class_conditional_score_matching_energy_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]:r["selected_frame"]=ch.copy();r["selected_summary"]=r["baseline_summary"];require(r["selected_frame"].equals(ch),"V198 fallback")
    return r
def material_gate(e):c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v)for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key");return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V198").resolve(),"binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate
    reports={"MODEL_COMPARISON.json":{"version":"V198","models":{"V69":e["baseline_summary"],"V198_DIAGNOSTIC":e["candidate_summary"],"V198_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V198","selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V198","selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V198_SCORE_MATCHING_ENERGY_REPORT.json":{"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V198_ENERGY_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V198_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full");return a
    return a
def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller deterministic CUDA0 DSM","gpu_model_calls":"class energy training"};authority=verify_authority();dev,ch=load_authorized()
    if a.full_run:require(os.environ.get("CUDA_VISIBLE_DEVICES")=="0"and torch.cuda.is_available(),"V198 full CUDA0 unavailable")
    if a.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"parameter_n":2*(INPUT_DIM*HIDDEN_DIM+HIDDEN_DIM+HIDDEN_DIM+1),"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"full_cuda0_fail_closed":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"nested_key":"selected.material_gate.nested_bootstrap","exact_fallback":True},"output_written":False}),indent=2));return
    with threadpool_limits(limits=2 if bounded else None):
        d,e,aud=run_nested(dev,ch,bounded,a.smoke_market)
        if a.support_probe:
            x=aud[0];non=[v for v in x["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":x["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":x["outer_baseline"],"raw_outer_selected":x["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":x["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":x["inner_chronology"],"strict_outer_chronology":x["outer_chronology"],"output_written":False}),indent=2));return
        ev=evaluate(ch,d,e,aud,250 if a.smoke_test else 2000,a.smoke_test)
    non=[v for v in aud[0]["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));s={"status":"SMOKE_OK"if a.smoke_test else ev["status"],"runtime":runtime,"selected_models":[x["policy"]["selected"]["name"]for x in aud],"outer_auc_delta":ev["outer_auc_delta"],"outer_ba_delta":ev["outer_ba_delta"],"outer_net_delta":ev["outer_net_delta"],"material_pass":ev["material_pass"],"fallback_is_exact_v69":not ev["material_pass"],"current_material_gate":material_gate(ev),"output_written":False}
    if a.smoke_test:s.update({"raw_inner_selected":aud[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":aud[0]["outer_baseline"],"raw_outer_selected":aud[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":aud[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(s),indent=2));return
    print(json.dumps(clean({**s,"output_written":True,"controller":write_outputs(a.output,authority,e,aud,ev)}),indent=2))
if __name__=="__main__":main()
