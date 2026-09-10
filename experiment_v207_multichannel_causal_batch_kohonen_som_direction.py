"""V207 label-free batch Kohonen SOM code-likelihood direction challenger."""
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
from threadpoolctl import threadpool_limits
import experiment_v198_multichannel_causal_score_matching_energy_direction as scaffold

core=scaffold.scaffold;base=scaffold.base;ROOT=scaffold.ROOT;VERSION=207;HYPOTHESIS="MULTICHANNEL_CAUSAL_BATCH_KOHONEN_SOM_DIRECTION_V1";DEFAULT_OUT=ROOT/"research"/"staging"/"V207"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v198_multichannel_causal_score_matching_energy_direction.py";EXPECTED_SCAFFOLD_SHA256="80338c3f473670813ee1a888958b44911d9e926ba469a455ed48b086514ef9bc"
STATE_DIM=16;GRID_SIDE=4;CODE_N=16;BATCH_EPOCHS=24;RADIUS_START=2.;RADIUS_END=.5;JEFFREYS=.5;SEED=20701;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"BATCH_KOHONEN_SOM_W0.25","weight":.25},{"name":"BATCH_KOHONEN_SOM_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
STATIC_SPEC={"feature_dimension":CODE_N,"state":"fixed nonoverlapping 2x2 average pool from 8x8 to 4x4/16D","representation":"label-free 4x4 toroidal batch Kohonen self-organizing map","initialization":"sixteen fixed chronologically equispaced strict-train group states","training":"24 deterministic batch updates with linear radius 2.0 to 0.5","head":"direct Jeffreys-smoothed class BMU code likelihood, no discriminative head","target_model_fit":False}

def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V198 scaffold changed");a=scaffold.verify_authority();a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V198 output"};a["v207_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return a
def compress_surface(path):return path.reshape(len(path),4,2,4,2).mean(axis=(2,4)).reshape(len(path),STATE_DIM)
def toroidal_distance2():
    position=np.array([(r,c)for r in range(GRID_SIDE)for c in range(GRID_SIDE)],float);delta=np.abs(position[:,None,:]-position[None,:,:]);delta=np.minimum(delta,GRID_SIDE-delta);return (delta*delta).sum(axis=2)
GRID_DISTANCE2=toroidal_distance2()
def bmu(state,codebook):return ((state[:,None,:]-codebook[None,:,:])**2).sum(axis=2).argmin(axis=1)
def fit_som(state):
    require(len(state)>=CODE_N,"V207 SOM support");index=np.rint(np.linspace(0,len(state)-1,CODE_N)).astype(int);codebook=state[index].copy();quantization=[]
    for epoch in range(BATCH_EPOCHS):
        winner=bmu(state,codebook);radius=RADIUS_START+(RADIUS_END-RADIUS_START)*epoch/(BATCH_EPOCHS-1);neighborhood=np.exp(-GRID_DISTANCE2[:,winner]/(2*radius*radius));den=neighborhood.sum(axis=1);require(np.all(den>1e-9),"V207 empty SOM unit");codebook=(neighborhood@state)/den[:,None];quantization.append(float(np.sqrt(np.min(((state[:,None,:]-codebook[None,:,:])**2).sum(axis=2),axis=1)).mean()))
    require(np.isfinite(codebook).all()and np.isfinite(quantization).all(),"V207 SOM finite");return codebook,{"epochs":BATCH_EPOCHS,"initial_index_sha256":array_sha256(index),"initial_quantization_error":quantization[0],"final_quantization_error":quantization[-1],"codebook_sha256":array_sha256(codebook)}
def code_probability(train_code,target_code,y):
    table=np.zeros((2,CODE_N),float)
    for cls in(0,1):
        counts=np.bincount(train_code[y==cls],minlength=CODE_N).astype(float)+JEFFREYS;table[cls]=counts/counts.sum()
    down=table[0,target_code];up=table[1,target_code];return np.clip(up/(up+down),1e-6,1-1e-6),table
def som_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);ts,ta=processor.transform(ordered);xs,xa=processor.transform(target);tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]);xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]);gp,y,grouping=equal_event_group_path(ordered,tp);scale=TrainChannelScale().fit(gp);train_x=compress_surface(scale.transform(gp));target_x=compress_surface(scale.transform(xp));codebook,training=fit_som(train_x);train_code=bmu(train_x,codebook);target_code=bmu(target_x,codebook);prob,table=code_probability(train_code,target_code,y)
    geometry={"rows":len(train_x),"feature_dimension":CODE_N,"label_free_batch_kohonen_som":True,"immutable_toroidal_code_lattice":True,"grid_side":GRID_SIDE,"code_n":CODE_N,"batch_epochs":BATCH_EPOCHS,"fixed_radius_schedule":[RADIUS_START,RADIUS_END],"train_only_codebook":True,"target_bmu_only":True,"jeffreys_class_code_likelihood":True,"model_fit_uses_target_rows":False,"target_labels_used":False,"training":training,"class_code_table_sha256":array_sha256(table),"label_independent_exact_transform":False,"local_differential_geometry_not_global_path_integral":False,"v97_nearest_analog_vote":False,"v108_archetype_simplex":False,"v114_nmf":False,"v123_gmm_fisher":False,"v206_capsule_routing":False}
    return prob,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":geometry,"target_geometry":{**geometry,"rows":len(target_x),"model_fit_uses_target_rows":False},"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":7.},"feature_scaling":{"train_only":True,"fixed_average_pool":True,"codebook_sha256":training["codebook_sha256"]},"head":{"type":"direct class BMU code likelihood","jeffreys":JEFFREYS},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"model_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(prob.mean()),"std":float(prob.std()),"probability_sha256":array_sha256(prob)}}

_BASE_BOOTSTRAP=scaffold._BASE_BOOTSTRAP
def configure_base():
    core.VERSION=VERSION;core.HYPOTHESIS=HYPOTHESIS;core.STATIC_SPEC=STATIC_SPEC;core.DAG_FEATURE_DIMENSION=CODE_N;core.ARCHITECTURES=ARCHITECTURES;core.SEED=SEED;core.dag_direction=som_direction;core.choose_inner=choose_inner;base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=CODE_N;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=som_direction
def choose_inner(tr,va,ch):
    challenger,a=som_direction(tr,va);b=ch.prob.to_numpy(float);conf=ch.confidence_signal.to_numpy(float);high=bool_series(ch.high_conf).to_numpy(bool);bm=metric(va,b,conf,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}];g=a["train_geometry"];ok=bool(g["label_free_batch_kohonen_som"]and g["immutable_toroidal_code_lattice"]and g["train_only_codebook"]and g["target_bmu_only"]and g["jeffreys_class_code_likelihood"]and not g["model_fit_uses_target_rows"]and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(b,challenger,ar["weight"]);m=metric(va,p,conf,high);auc=m["auc"]-bm["auc"];ba=m["balanced_accuracy"]-bm["balanced_accuracy"];net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"];floor,d=base.supported_source_ba_delta(va,b,p);trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    s=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return{"selection_rule":"inner OOF V69 noop or fixed .25/.50 batch Kohonen SOM code-likelihood blend","baseline":bm,"selected":s,"trials":trials,"outer_labels_used_for_selection":False,"pool_grid_initialization_epoch_radius_smoothing_blend_or_safety_micro_tuning":False},a
def run_nested(dev,ch,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):d,e,a=core.run_nested(dev,ch,smoke,smoke_market)
    d=d.rename(columns={"v195_model":"v207_model"});e=e.rename(columns={"sparse_dag_moment_prob":"batch_kohonen_som_prob","v195_model":"v207_model"})
    for x in a:
        s=x["policy"]["selected"];print(f"[V207 BATCH SOM] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws):configure_base();r=_BASE_BOOTSTRAP(e,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V207 batch Kohonen SOM policy";r["seed"]=SEED;r["standardized"]=True;return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(ch,d,e,a,draws,smoke);c=r["material_checks"];c.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(x[side]["train_geometry"]["label_free_batch_kohonen_som"]and x[side]["train_geometry"]["immutable_toroidal_code_lattice"]and x[side]["train_geometry"]["target_bmu_only"]and not x[side]["model_fit_uses_target_rows"]and not x[side]["target_labels_used"]for x in a for side in("inner_model_audit","outer_model_audit"));c["multichannel_causal_batch_kohonen_som_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]:r["selected_frame"]=ch.copy();r["selected_summary"]=r["baseline_summary"];require(r["selected_frame"].equals(ch),"V207 fallback")
    return r
def material_gate(e):c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v)for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key");return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V207").resolve(),"binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate
    reports={"MODEL_COMPARISON.json":{"version":"V207","models":{"V69":e["baseline_summary"],"V207_DIAGNOSTIC":e["candidate_summary"],"V207_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V207","selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V207","selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V207_BATCH_KOHONEN_SOM_REPORT.json":{"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V207_BATCH_SOM_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V207_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full");return a
    return a
def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller deterministic CPU batch Kohonen SOM","gpu_model_calls":0};authority=verify_authority();dev,ch=load_authorized()
    if a.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"noarg":True,"conflict":True,"direct_full":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"nested_key":"selected.material_gate.nested_bootstrap","exact_fallback":True},"output_written":False}),indent=2));return
    with threadpool_limits(limits=2 if bounded else None):
        d,e,aud=run_nested(dev,ch,bounded,a.smoke_market)
        if a.support_probe:
            x=aud[0];non=[v for v in x["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":x["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":x["outer_baseline"],"raw_outer_selected":x["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":x["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":x["inner_chronology"],"strict_outer_chronology":x["outer_chronology"],"output_written":False}),indent=2));return
        ev=evaluate(ch,d,e,aud,250 if a.smoke_test else 2000,a.smoke_test)
    non=[v for v in aud[0]["policy"]["trials"]if v["architecture"]];best=max(non,key=lambda v:(v["eligible"],v["score"]));s={"status":"SMOKE_OK"if a.smoke_test else ev["status"],"runtime":runtime,"selected_models":[x["policy"]["selected"]["name"]for x in aud],"outer_auc_delta":ev["outer_auc_delta"],"outer_ba_delta":ev["outer_ba_delta"],"outer_net_delta":ev["outer_net_delta"],"material_pass":ev["material_pass"],"fallback_is_exact_v69":not ev["material_pass"],"current_material_gate":material_gate(ev),"output_written":False}
    if a.smoke_test:s.update({"raw_inner_selected":aud[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":aud[0]["outer_baseline"],"raw_outer_selected":aud[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":aud[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(s),indent=2));return
    print(json.dumps(clean({**s,"output_written":True,"controller":write_outputs(a.output,authority,e,aud,ev)}),indent=2))
if __name__=="__main__":main()
