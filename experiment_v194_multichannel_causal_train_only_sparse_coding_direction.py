"""V194 train-only patch sparse-coding direction challenger.

Every strict-past train-scaled observed 8x8 causal surface is split into four
immutable 4x4 quadrant patches.  A deterministic 16-atom mini-batch dictionary
is fitted on equal-event-group training patches only.  Frozen OMP-4 codes and
fixed quadrant/atom summaries feed one balanced ridge direction head.  Target
patches only transform through the frozen dictionary; they never fit it.
"""
from __future__ import annotations

import os
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"): os.environ[_name]="2"

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v191_multichannel_causal_fixed_random_convolutional_scattering_reservoir_direction as scaffold

base=scaffold.base;ROOT=scaffold.ROOT;VERSION=194
HYPOTHESIS="MULTICHANNEL_CAUSAL_TRAIN_ONLY_SPARSE_CODING_DIRECTION_V1"
DEFAULT_OUT=ROOT/"research"/"staging"/"V194"/"LOCAL_FORBIDDEN"
FEATURES=scaffold.FEATURES;EXPECTED_V69_EXPERIMENT_ID=scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH=ROOT/"experiment_v191_multichannel_causal_fixed_random_convolutional_scattering_reservoir_direction.py"
EXPECTED_SCAFFOLD_SHA256="b28f4a06317180b8ef9c15c67c2f040c81402aaee3740152c19313b3c0039906"
ATOM_N=16;OMP_NONZERO=4;DICTIONARY_ITERATIONS=32;PATCH_N=4;PATCH_DIM=16
SPARSE_FEATURE_DIMENSION=PATCH_N*ATOM_N*2+ATOM_N*3+PATCH_N*2
SEED=19401;RIDGE_LOGISTIC_C=.5;FEATURE_CLIP=8.;CHANNEL_CLIP=7.;BOOTSTRAP_DRAWS=2000;SMOKE_BOOTSTRAP_DRAWS=250;SMOKE_THREADS=2
ARCHITECTURES=({"name":"TRAIN_ONLY_SPARSE_CODING_W0.25","weight":.25},{"name":"TRAIN_ONLY_SPARSE_CODING_W0.50","weight":.50})

require=scaffold.require;sha256=scaffold.sha256;array_sha256=scaffold.array_sha256;clean=scaffold.clean;bool_series=scaffold.bool_series
atomic_json=scaffold.atomic_json;atomic_csv=scaffold.atomic_csv;load_authorized=scaffold.load_authorized;metric=scaffold.metric
controller=scaffold.controller;numeric=scaffold.numeric;observed_past_to_recent_path=scaffold.observed_past_to_recent_path
equal_event_group_path=scaffold.equal_event_group_path;TrainChannelScale=scaffold.TrainChannelScale;blend_probability=scaffold.blend_probability

STATIC_SPEC={"path_shape":[8,8],"patches":"four immutable old/recent by low/high-channel 4x4 quadrants","dictionary":"strict-train-only deterministic 16-atom MiniBatchDictionaryLearning, 32 iterations, shuffle false","coding":"frozen orthogonal matching pursuit with exactly four nonzero atoms","feature":"signed and absolute quadrant codes, atom pool moments and patch mean/std","feature_dimension":SPARSE_FEATURE_DIMENSION,"head":"equal-event-group/class-balanced C=0.5 ridge logistic","target_dictionary_fit":False}
require(SPARSE_FEATURE_DIMENSION==184,"V194 sparse feature dimension changed")


class RobustFeatureScale:
    def fit(self,x:np.ndarray)->"RobustFeatureScale":
        require(x.ndim==2 and x.shape[1]==SPARSE_FEATURE_DIMENSION,"V194 feature shape invalid");self.median=np.median(x,axis=0);q25,q75=np.quantile(x,[.25,.75],axis=0);robust=(q75-q25)/1.349;standard=np.std(x,axis=0);self.scale=np.where(robust>1e-8,robust,np.where(standard>1e-8,standard,1.));return self
    def transform(self,x:np.ndarray)->np.ndarray:
        result=np.clip((x-self.median)/self.scale,-FEATURE_CLIP,FEATURE_CLIP);require(result.shape[1]==SPARSE_FEATURE_DIMENSION and np.isfinite(result).all(),"V194 scaled feature invalid");return result


def verify_authority()->dict[str,Any]:
    require(sha256(SCAFFOLD_PATH)==EXPECTED_SCAFFOLD_SHA256,"pinned V191 scaffold changed");audit=scaffold.verify_authority();audit["code_dependency"]={"path":SCAFFOLD_PATH.name,"sha256":EXPECTED_SCAFFOLD_SHA256,"purpose":"V36/V69 authority, chronology, path, metrics, gate/bootstrap/atomic IO only; no V191 output is read"};audit["v194_access_contract"]={"immutable_v36_dev_only":True,"atomic_output_v69_only":True,"failed_outputs_read":False,"dev_extension_read":False,"role_assignment_read":False,"research_seal_read":False,"final_reserve_read":False,"prior_version_output_read":False};return audit


def quadrant_patches(path:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    require(path.ndim==3 and path.shape[1:]==(8,8),"V194 path shape invalid")
    patches=np.stack([path[:,:4,:4],path[:,:4,4:],path[:,4:,:4],path[:,4:,4:]],axis=1).reshape(len(path),PATCH_N,PATCH_DIM)
    mean=patches.mean(axis=2);scale=patches.std(axis=2);normalized=(patches-mean[:,:,None])/np.maximum(scale[:,:,None],1e-6)
    return normalized,mean,scale


def sparse_features(path:np.ndarray,dictionary:MiniBatchDictionaryLearning|None=None,fit:bool=False)->tuple[np.ndarray,MiniBatchDictionaryLearning,dict[str,Any]]:
    patches,patch_mean,patch_scale=quadrant_patches(path);flat=patches.reshape(-1,PATCH_DIM)
    if fit:
        require(dictionary is None,"V194 dictionary supplied during fit");dictionary=MiniBatchDictionaryLearning(n_components=ATOM_N,alpha=1.0,max_iter=DICTIONARY_ITERATIONS,batch_size=256,shuffle=False,transform_algorithm="omp",transform_n_nonzero_coefs=OMP_NONZERO,random_state=SEED,verbose=False);dictionary.fit(flat)
    require(dictionary is not None,"V194 dictionary missing");codes=dictionary.transform(flat).reshape(len(path),PATCH_N,ATOM_N)
    signed=codes.reshape(len(path),-1);absolute=np.abs(codes).reshape(len(path),-1);atom=[codes.mean(axis=1),codes.std(axis=1),np.abs(codes).max(axis=1)]
    feature=np.concatenate([signed,absolute]+atom+[patch_mean,patch_scale],axis=1)
    nonzero=np.sum(np.abs(codes)>1e-12,axis=2);require(feature.shape==(len(path),SPARSE_FEATURE_DIMENSION) and np.isfinite(feature).all(),"V194 sparse feature invalid");require(int(nonzero.max())<=OMP_NONZERO,"V194 OMP support exceeded")
    audit={"rows":len(path),"feature_dimension":feature.shape[1],"feature_sha256":array_sha256(feature),"dictionary_fit_on_this_frame":bool(fit),"dictionary_components_sha256":array_sha256(dictionary.components_),"dictionary_atom_n":ATOM_N,"dictionary_iterations":DICTIONARY_ITERATIONS,"omp_nonzero":OMP_NONZERO,"mean_nonzero_codes":float(nonzero.mean()),"strict_train_only_dictionary":True,"target_dictionary_fit":False,"supervised_dictionary_or_target_label_use":False,"v114_nonnegative_matrix_factorization":False,"v105_fastica":False,"v108_archetype_simplex":False,"v82_masked_autoencoder":False,"v191_frozen_random_convolution":False,"label_independent_exact_transform":True,"local_differential_geometry_not_global_path_integral":True}
    return feature,dictionary,audit


def sparse_direction(train:pd.DataFrame,target_frame:pd.DataFrame)->tuple[np.ndarray,dict[str,Any]]:
    ordered=train.sort_values(["event_time_utc","event_id"],kind="stable").reset_index(drop=True);require(ordered.market.nunique()==target_frame.market.nunique()==1,"V194 expects one market")
    processor=numeric.RobustNumericState().fit(ordered);train_state,train_transform=processor.transform(ordered);target_state,target_transform=processor.transform(target_frame)
    train_path,train_path_audit=observed_past_to_recent_path(train_state[:,:len(FEATURES)]);target_path,target_path_audit=observed_past_to_recent_path(target_state[:,:len(FEATURES)])
    group_path,group_target,grouping=equal_event_group_path(ordered,train_path);channel_scale=TrainChannelScale().fit(group_path);group_path=channel_scale.transform(group_path);target_path=channel_scale.transform(target_path)
    train_feature,dictionary,train_sparse=sparse_features(group_path,fit=True);target_feature,_,target_sparse=sparse_features(target_path,dictionary=dictionary,fit=False)
    feature_scale=RobustFeatureScale().fit(train_feature);train_design=feature_scale.transform(train_feature);target_design=feature_scale.transform(target_feature)
    class_n=np.bincount(group_target,minlength=2).astype(float);require(np.all(class_n>0),"V194 train class missing");weight=np.where(group_target==1,.5/class_n[1],.5/class_n[0])
    head=LogisticRegression(C=RIDGE_LOGISTIC_C,penalty="l2",solver="liblinear",max_iter=1000,random_state=SEED);head.fit(train_design,group_target,sample_weight=weight);prob=np.clip(head.predict_proba(target_design)[:,1],1e-6,1-1e-6)
    return prob,{"train_n":len(ordered),"train_event_group_n":len(group_target),"target_n":len(target_frame),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":train_transform,"target_transform":target_transform,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":train_path_audit,"target_path":target_path_audit,"train_geometry":train_sparse,"target_geometry":target_sparse,"channel_scaling":{"train_only":True,"median_sha256":array_sha256(channel_scale.median),"scale_sha256":array_sha256(channel_scale.scale),"clip":CHANNEL_CLIP},"feature_scaling":{"train_only":True,"median_sha256":array_sha256(feature_scale.median),"scale_sha256":array_sha256(feature_scale.scale),"clip":FEATURE_CLIP},"head":{"type":"balanced ridge logistic","C":RIDGE_LOGISTIC_C,"coefficient_sha256":array_sha256(head.coef_),"iterations":int(head.n_iter_[0])},"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"dictionary_fit_uses_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(prob.mean()),"std":float(prob.std()),"probability_sha256":array_sha256(prob)}}


_BASE_BOOTSTRAP=base.paired_nested_bootstrap
def configure_base()->None:
    scaffold.VERSION=VERSION;scaffold.HYPOTHESIS=HYPOTHESIS;scaffold.STATIC_SPEC=STATIC_SPEC;scaffold.RESERVOIR_FEATURE_DIMENSION=SPARSE_FEATURE_DIMENSION;scaffold.ARCHITECTURES=ARCHITECTURES;scaffold.SEED=SEED;scaffold.reservoir_direction=sparse_direction;scaffold.choose_inner=choose_inner;base.VERSION=VERSION;base.HYPOTHESIS=HYPOTHESIS;base.STATIC_SPEC=STATIC_SPEC;base.FRENET_FEATURE_DIMENSION=SPARSE_FEATURE_DIMENSION;base.ARCHITECTURES=ARCHITECTURES;base.SEED=SEED;base.frenet_direction=sparse_direction


def choose_inner(inner_train:pd.DataFrame,inner_valid:pd.DataFrame,inner_champion:pd.DataFrame)->tuple[dict[str,Any],dict[str,Any]]:
    challenger,audit=sparse_direction(inner_train,inner_valid);baseline=inner_champion.prob.to_numpy(float);confidence=inner_champion.confidence_signal.to_numpy(float);high=bool_series(inner_champion.high_conf).to_numpy(bool);baseline_metric=metric(inner_valid,baseline,confidence,high)
    trials=[{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.,"balanced_accuracy_delta":0.,"all_trade_net_delta":0.,"worst_supported_source_ba_delta":0.,"supported_source_ba_delta":{},"eligible":True,"score":2*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}]
    geometry=audit["train_geometry"];eligible=bool(audit["causal_numeric_only"] and audit["grouping"]["equal_event_group_weighting"] and geometry["dictionary_fit_on_this_frame"] and geometry["strict_train_only_dictionary"] and not audit["target_geometry"]["dictionary_fit_on_this_frame"] and not audit["dictionary_fit_uses_target_rows"] and audit["channel_scaling"]["train_only"] and audit["feature_scaling"]["train_only"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]);current=metric(inner_valid,probability,confidence,high);auc=current["auc"]-baseline_metric["auc"];ba=current["balanced_accuracy"]-baseline_metric["balanced_accuracy"];net=current["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"];floor,deltas=base.supported_source_ba_delta(inner_valid,baseline,probability);trials.append({"name":architecture["name"],"architecture":architecture,"metrics":current,"auc_delta":auc,"balanced_accuracy_delta":ba,"all_trade_net_delta":net,"worst_supported_source_ba_delta":floor,"supported_source_ba_delta":deltas,"model_eligible":eligible,"eligible":bool(eligible and ba>=-.005 and net>=-.0005 and floor>=-.010),"score":2*current["auc"]+current["balanced_accuracy"]})
    selected=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return {"selection_rule":"inner OOF only; V69 noop or fixed .25/.50 train-only sparse-code blend","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"atom_iteration_sparsity_blend_or_safety_micro_tuning":False},audit


def run_nested(dev:pd.DataFrame,champion:pd.DataFrame,smoke:bool,smoke_market:str="US"):
    configure_base()
    with redirect_stdout(StringIO()):diagnostic,evidence,audits=scaffold.run_nested(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v191_model":"v194_model"});evidence=evidence.rename(columns={"random_conv_scattering_prob":"sparse_coding_prob","v191_model":"v194_model"})
    for audit in audits:
        s=audit["policy"]["selected"];print(f"[V194 SPARSE CODING] market={audit['market']} fold={audit['fold']} selected={s['name']} inner_auc_delta={s['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}",flush=True)
    return diagnostic,evidence,audits


def paired_nested_bootstrap(evidence:pd.DataFrame,draws:int)->dict[str,Any]:
    configure_base();r=_BASE_BOOTSTRAP(evidence,draws);r["method"]="paired market-fold-time block bootstrap over inner-locked V194 train-only sparse-coding policy";r["seed"]=SEED;r["standardized"]=True;return r


def evaluate(champion,diagnostic,evidence,audits,draws,smoke):
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;r=base.evaluate(champion,diagnostic,evidence,audits,draws,smoke);checks=r["material_checks"];checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract=all(a[side]["causal_numeric_only"] and a[side]["static_spec"]["feature_dimension"]==SPARSE_FEATURE_DIMENSION and a[side]["train_geometry"]["dictionary_fit_on_this_frame"] and not a[side]["target_geometry"]["dictionary_fit_on_this_frame"] and a[side]["train_geometry"]["strict_train_only_dictionary"] and not a[side]["dictionary_fit_uses_target_rows"] and a[side]["channel_scaling"]["train_only"] and a[side]["feature_scaling"]["train_only"] and not a[side]["target_labels_used"] for a in audits for side in ("inner_model_audit","outer_model_audit"));checks["multichannel_causal_train_only_sparse_coding_contract_verified"]=bool(contract);r["material_pass"]=bool(not smoke and all(checks.values()))
    if not r["material_pass"]:r["selected_frame"]=champion.copy();r["selected_summary"]=r["baseline_summary"];r["fallback"]={"activated":True,"policy":"exact entire V69 DataFrame","exact_entire_frame_verified":bool(r["selected_frame"].equals(champion))};require(r["selected_frame"].equals(champion),"V194 fallback not exact")
    return r


def material_gate(e):
    c=e["material_checks"];g={"contract":HYPOTHESIS,"checks":c,"passed":sum(bool(x)for x in c.values()),"total":len(c),"material_pass":bool(e["material_pass"]),"nested_bootstrap":e["nested_bootstrap"],"native_robustness_bootstrap":e["candidate_native_robustness_bootstrap"],"current_hypothesis_not_inherited_from_v69":True,"fallback_policy":"candidate"if e["material_pass"]else"exact entire V69 DataFrame"};require(g["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V194 nested key");return g


def enforce_output_conflict_fail_closed(out:Path):
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V194 output binding");require(out.resolve().parent==(ROOT/"research"/"staging"/"V194").resolve(),"V194 staging");require(out.exists()and out.is_dir(),"V194 target");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing)and not(existing-allowed),"V194 prefile conflict");return {"controller_bound":True,"required_controller_prefiles":sorted(required),"unexpected_prefiles":[]}


def write_outputs(out,authority,evidence,audits,e):
    conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in e.items()if not k.endswith("_frame")};gate=material_gate(e);selected=json.loads(json.dumps(clean(e["selected_summary"])));selected["material_gate"]=gate;require("bootstrap"in selected["robustness"],"native bootstrap")
    reports={"MODEL_COMPARISON.json":{"version":"V194","hypothesis":HYPOTHESIS,"status":e["status"],"models":{"V69_CHAMPION":e["baseline_summary"],"V194_DIAGNOSTIC_SPARSE_CODING":e["candidate_summary"],"V194_FAIL_CLOSED_SELECTED":selected},"evaluation":compact,"authority_audit":authority,"nested_fold_audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V194","status":e["status"],"selected":selected,"candidate":e["candidate_summary"],"material_gate":gate,"nested_bootstrap":e["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not e["material_pass"],"seal_authorized":False},"SOURCE_TRANSFER_REPORT.json":{"version":"V194","status":e["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":e["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":e["candidate_summary"]["metrics"]["by_market"],"material_gate":gate,"nested_bootstrap":e["nested_bootstrap"],"seal_authorized":False},"V194_SPARSE_CODING_REPORT.json":{"version":"V194","static_spec":STATIC_SPEC,"audits":audits,"evaluation":compact,"authority":authority},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"version":"V194","contract_key":"selected.material_gate.nested_bootstrap","bootstrap":e["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"version":"V194","bootstrap":e["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"version":"V194","status":"MATCH","canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected),"material_gate":gate},"RUN_STATUS.json":{"version":VERSION,"status":e["status"],"material_pass":e["material_pass"],"seal_authorized":False,"output_conflict_audit":conflict}}
    for n,p in reports.items():atomic_json(p,out/n)
    atomic_csv(evidence,out/"V194_SPARSE_CODING_OUTER_EVIDENCE.csv.gz");atomic_csv(e["selected_frame"],out/"V194_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in out.iterdir()if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"hypothesis":HYPOTHESIS,"files":files},out/"ARTIFACT_MANIFEST.json");return {"artifact_count":len(files)+1}


def parse_args():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));m.add_argument("--audit-only",action="store_true");m.add_argument("--smoke-test",action="store_true");m.add_argument("--support-probe",action="store_true");m.add_argument("--full-run",action="store_true");p.add_argument("--smoke-market",choices=("US","KR"),default="US");p.add_argument("--output",type=Path);a=p.parse_args()
    if CONTROLLER_OUTPUT:require(not(a.audit_only or a.smoke_test or a.support_probe or a.full_run or a.output),"V194 explicit conflict");a.full_run=True;a.output=Path(CONTROLLER_OUTPUT)
    elif a.output is None:a.output=DEFAULT_OUT
    if a.full_run:require(bool(CONTROLLER_OUTPUT)and a.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V194 direct full forbidden")
    return a


def main():
    a=parse_args();bounded=bool(a.audit_only or a.smoke_test or a.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller full CPU deterministic sparse dictionary","gpu_model_calls":0};authority=verify_authority();dev,champion=load_authorized()
    if a.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"static_spec":STATIC_SPEC,"canonical_controller_gate_checks":14,"controller_contract":{"no_argument_exact_binding":True,"explicit_conflict_fail_closed":True,"direct_full_fail_closed":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"nested_bootstrap_key":"selected.material_gate.nested_bootstrap","exact_entire_v69_fallback":True},"output_written":False}),indent=2));return
    with threadpool_limits(limits=2 if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,bounded,a.smoke_market)
        if a.support_probe:
            require(a.smoke_market=="KR","V194 KR support only");audit=audits[0];non=[x for x in audit["policy"]["trials"]if x["architecture"]];best=max(non,key=lambda x:(x["eligible"],x["score"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","runtime":runtime,"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"output_written":False}),indent=2));return
        e=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if a.smoke_test else BOOTSTRAP_DRAWS,a.smoke_test)
    non=[x for x in audits[0]["policy"]["trials"]if x["architecture"]];best=max(non,key=lambda x:(x["eligible"],x["score"]));summary={"status":"SMOKE_OK"if a.smoke_test else e["status"],"runtime":runtime,"selected_models":[x["policy"]["selected"]["name"]for x in audits],"outer_auc_delta":e["outer_auc_delta"],"outer_ba_delta":e["outer_ba_delta"],"outer_net_delta":e["outer_net_delta"],"material_pass":e["material_pass"],"fallback_is_exact_v69":not e["material_pass"],"current_material_gate":material_gate(e),"output_written":False}
    if a.smoke_test:summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]});print(json.dumps(clean(summary),indent=2));return
    result=write_outputs(a.output,authority,evidence,audits,e);print(json.dumps(clean({**summary,"output_written":True,"controller":result}),indent=2))

if __name__=="__main__":main()
