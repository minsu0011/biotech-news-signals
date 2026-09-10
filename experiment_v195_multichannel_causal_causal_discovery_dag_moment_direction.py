"""V195 strict-train-only fixed-order sparse Cholesky-DAG moment direction."""
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
from sklearn.linear_model import Lasso,LogisticRegression
from threadpoolctl import threadpool_limits
import experiment_v194_multichannel_causal_train_only_sparse_coding_direction as scaffold

base=scaffold.base;ROOT=scaffold.ROOT;VERSION=195;HYPOTHESIS="MULTICHANNEL_CAUSAL_CAUSAL_DISCOVERY_DAG_MOMENT_DIRECTION_V1"
DEFAULT_OUT=ROOT/"research"/"staging"/"V195"/"LOCAL_FORBIDDEN";FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v194_multichannel_causal_train_only_sparse_coding_direction.py";EXPECTED_SCAFFOLD_SHA256="b4cefa0b3dabed24627fea36271aeb9fea94ba057f0cb7fba344b8a41f4b850f"
SITE_N=64;HORIZON_N=8;CHANNEL_N=8;LASSO_ALPHA=.05;MAX_ITER=2000;DAG_FEATURE_DIMENSION=192;SEED=19501;RIDGE_LOGISTIC_C=.5;FEATURE_CLIP=8.;CHANNEL_CLIP=7.;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250
ARCHITECTURES=({"name":"TRAIN_ONLY_SPARSE_DAG_MOMENT_W0.25","weight":.25},{"name":"TRAIN_ONLY_SPARSE_DAG_MOMENT_W0.50","weight":.50})
require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series;atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric;controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path;equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability
STATIC_SPEC={"path_shape":[8,8],"node_order":"horizon-major strict past-to-recent; no same-horizon or future parent edges","graph_fit":"64 deterministic Lasso structural equations, alpha .05, earlier-horizon parents only, strict-train-only label-free","features":"direct SEM prediction, innovation residual and finite ancestor total-effect drive pooled by horizon and channel","feature_dimension":DAG_FEATURE_DIMENSION,"head":"balanced C=.5 ridge logistic","target_graph_fit":False}

class RobustFeatureScale:
    def fit(self,x):
        require(x.shape[1]==DAG_FEATURE_DIMENSION,"V195 dim");self.median=np.median(x,0);q25,q75=np.quantile(x,[.25,.75],axis=0);r=(q75-q25)/1.349;s=np.std(x,0);self.scale=np.where(r>1e-8,r,np.where(s>1e-8,s,1.));return self
    def transform(self,x):
        z=np.clip((x-self.median)/self.scale,-FEATURE_CLIP,FEATURE_CLIP);require(np.isfinite(z).all(),"V195 scaled");return z

def verify_authority():
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"V194 scaffold changed");a=scaffold.verify_authority();a["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"authority/chronology/path/metrics/gate/bootstrap/IO only; no V194 output"};a["v195_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return a

def fit_sparse_dag(path:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    x=path.reshape(len(path),SITE_N);b=np.zeros((SITE_N,SITE_N));intercept=np.zeros(SITE_N)
    for h in range(1,HORIZON_N):
        parents=np.arange(h*CHANNEL_N)
        for c in range(CHANNEL_N):
            j=h*CHANNEL_N+c;model=Lasso(alpha=LASSO_ALPHA,fit_intercept=True,max_iter=MAX_ITER,tol=1e-5,selection="cyclic",random_state=SEED);model.fit(x[:,parents],x[:,j]);b[parents,j]=model.coef_;intercept[j]=model.intercept_
    require(np.allclose(np.triu(b.T,0),np.zeros_like(b)),"V195 DAG order violation")
    return b,intercept

def dag_features(path:np.ndarray,b:np.ndarray,intercept:np.ndarray)->tuple[np.ndarray,dict[str,Any]]:
    x=path.reshape(len(path),SITE_N);pred=x@b+intercept;resid=x-pred;total=np.eye(SITE_N);power=np.eye(SITE_N)
    for _ in range(HORIZON_N-1):power=power@b;total+=power
    drive=x@(total-np.eye(SITE_N));fields=[pred.reshape(-1,8,8),resid.reshape(-1,8,8),drive.reshape(-1,8,8)];blocks=[]
    for field in fields:
        blocks.extend([field.mean(2),field.std(2),field.max(2),field.min(2),field.mean(1),field.std(1),field.max(1),field.min(1)])
    z=np.concatenate(blocks,1);require(z.shape==(len(path),DAG_FEATURE_DIMENSION)and np.isfinite(z).all(),"V195 feature")
    edges=np.abs(b)>1e-12;return z,{"rows":len(path),"feature_dimension":z.shape[1],"feature_sha256":array_sha256(z),"edge_n":int(edges.sum()),"edge_density":float(edges.mean()),"dag_sha256":array_sha256(b),"strict_train_only_graph":True,"target_graph_fit":False,"label_free_graph_fit":True,"earlier_horizon_parents_only":True,"same_or_future_horizon_edges":False,"acyclic_by_construction":True,"fixed_lasso_alpha":LASSO_ALPHA,"v90_tan_class_graph":False,"v106_handwired_feature_graph":False,"v119_dmd_transition":False,"v159_transfer_entropy":False,"granger_or_var_forecast":False,"label_independent_exact_transform":True,"local_differential_geometry_not_global_path_integral":True}

def dag_direction(train,target):
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);processor=numeric.RobustNumericState().fit(ordered);ts,ta=processor.transform(ordered);xs,xa=processor.transform(target);tp,tpa=observed_past_to_recent_path(ts[:,:len(FEATURES)]);xp,xpa=observed_past_to_recent_path(xs[:,:len(FEATURES)]);gp,y,grouping=equal_event_group_path(ordered,tp);scale=TrainChannelScale().fit(gp);gp=scale.transform(gp);xp=scale.transform(xp);b,i=fit_sparse_dag(gp);tf,tg=dag_features(gp,b,i);xf,xg=dag_features(xp,b,i);fs=RobustFeatureScale().fit(tf);td=fs.transform(tf);xd=fs.transform(xf);n=np.bincount(y,minlength=2).astype(float);w=np.where(y==1,.5/n[1],.5/n[0]);head=LogisticRegression(C=.5,solver="liblinear",max_iter=1000,random_state=SEED);head.fit(td,y,sample_weight=w);p=np.clip(head.predict_proba(xd)[:,1],1e-6,1-1e-6)
    return p,{"train_n":len(ordered),"train_event_group_n":len(y),"target_n":len(target),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":ta,"target_transform":xa,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":tpa,"target_path":xpa,"train_geometry":tg,"target_geometry":xg,"channel_scaling":{"train_only":True,"median_sha256":array_sha256(scale.median),"scale_sha256":array_sha256(scale.scale),"clip":CHANNEL_CLIP},"feature_scaling":{"train_only":True,"median_sha256":array_sha256(fs.median),"scale_sha256":array_sha256(fs.scale),"clip":FEATURE_CLIP},"head":{"type":"balanced ridge","C":.5,"coefficient_sha256":array_sha256(head.coef_)},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"graph_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(p.mean()),"std":float(p.std()),"probability_sha256":array_sha256(p)}}

_BASE_BOOTSTRAP=base.paired_nested_bootstrap
def configure_base():
    scaffold.VERSION=VERSION;scaffold.HYPOTHESIS=HYPOTHESIS;scaffold.STATIC_SPEC=STATIC_SPEC;scaffold.SPARSE_FEATURE_DIMENSION=DAG_FEATURE_DIMENSION;scaffold.ARCHITECTURES=ARCHITECTURES;scaffold.SEED=SEED;scaffold.sparse_direction=dag_direction;scaffold.choose_inner=choose_inner;base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=DAG_FEATURE_DIMENSION;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=dag_direction
def choose_inner(tr,va,ch):
    challenger,a=dag_direction(tr,va);baseline=ch.prob.to_numpy(float);conf=ch.confidence_signal.to_numpy(float);high=bool_series(ch.high_conf).to_numpy(bool);bm=metric(va,baseline,conf,high);trials=[{"name":"V69_NOOP","architecture":None,"metrics":bm,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*bm["auc"]+bm["balanced_accuracy"]}];g=a["train_geometry"];ok=bool(g["strict_train_only_graph"]and g["label_free_graph_fit"]and g["acyclic_by_construction"]and not a["target_geometry"]["target_graph_fit"]and not a["graph_fit_uses_target_rows"]and not a["target_labels_used"])
    for ar in ARCHITECTURES:
        p=blend_probability(baseline,challenger,ar["weight"]);m=metric(va,p,conf,high);auc=m["auc"]-bm["auc"];ba=m["balanced_accuracy"]-bm["balanced_accuracy"];net=m["all_trade_mean_signed_net"]-bm["all_trade_mean_signed_net"];floor,d=base.supported_source_ba_delta(va,baseline,p);trials.append({"name":ar["name"],"architecture":ar,"metrics":m,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":d,"model_eligible":ok,"eligible":bool(ok and ba>=-.005 and net>=-.0005 and floor>=-.01),"score":2*m["auc"]+m["balanced_accuracy"]})
    s=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return {"selection_rule":"inner OOF only V69/noop or fixed .25/.50 sparse-DAG moment blend","baseline":bm,"selected":s,"trials":trials,"outer_labels_used_for_selection":False,"alpha_order_pool_blend_or_safety_micro_tuning":False},a
def run_nested(dev,champion,smoke,smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):d,e,a=scaffold.run_nested(dev,champion,smoke,smoke_market)
    d=d.rename(columns={"v194_model":"v195_model"});e=e.rename(columns={"sparse_coding_prob":"sparse_dag_moment_prob","v194_model":"v195_model"})
    for x in a:
        s=x["policy"]["selected"];print(f"[V195 SPARSE DAG] market={x['market']} fold={x['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={x['outer_auc_delta']:+.6f}")
    return d,e,a
def paired_nested_bootstrap(e,draws):configure_base();r=_BASE_BOOTSTRAP(e,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V195 sparse-DAG moment policy";r["seed"]=SEED;r["standardized"]=True;return r
def evaluate(ch,d,e,a,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(ch,d,e,a,draws,smoke);c=r["material_checks"];c.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=all(x[side]["train_geometry"]["strict_train_only_graph"]and x[side]["train_geometry"]["acyclic_by_construction"]and not x[side]["target_geometry"]["target_graph_fit"]and not x[side]["graph_fit_uses_target_rows"]and not x[side]["target_labels_used"]for x in a for side in("inner_model_audit","outer_model_audit"));c["multichannel_causal_train_only_sparse_cholesky_dag_moment_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(c.values()))
    if not r["material_pass"]:r["selected_frame"]=ch.copy();r["selected_summary"]=r["baseline_summary"];require(r["selected_frame"].equals(ch),"V195 fallback")
    return r
def material_gate(e):c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(v)for v in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","key");return g
def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve()and out.resolve().parent==(ROOT/"research"/"staging"/"V195").resolve(),"binding");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"conflict");return{"controller_bound":True}
def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate
    reports={"MODEL_COMPARISON.json":{"version":"V195","models":{"V69_CHAMPION":e["baseline_summary"],"V195_DIAGNOSTIC":e["candidate_summary"],"V195_SELECTED":selected},"evaluation":compact,"authority":authority,"audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V195","selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"fallback_is_exact_entire_v69_frame":not e["material_pass"]},"SOURCE_TRANSFER_REPORT.json":{"version":"V195","selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate},"V195_DAG_REPORT.json":{"static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected)},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"conflict":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V195_DAG_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V195_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"files":files},out/"ARTIFACT_MANIFEST.json");return{"artifact_count":len(files)+1}
def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"direct full");return a
    return a
def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller CPU deterministic sparse DAG","gpu_model_calls":0};authority=verify_authority();dev,ch=load_authorized()
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
