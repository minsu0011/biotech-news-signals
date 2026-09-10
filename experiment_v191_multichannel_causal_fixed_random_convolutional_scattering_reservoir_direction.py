"""V191 fixed random convolutional scattering-reservoir direction challenger.

The requested masked autoencoder was rejected because V82 already implements
masked self-supervised state reconstruction.  V191 instead maps each strict-
past train-scaled observed 8x8 causal surface through two immutable seeded 3x3
random convolution banks with absolute-tanh scattering nonlinearities.  Fixed
regional moments of both layers and raw-channel summaries feed one equal-event-
group/class-balanced ridge-logit direction head.  Filters are never trained.

Bounded audit/support uses dependency-free NumPy on CPU30-31.  A controller-
authorized full run requires runtime_limits CUDA0 and executes the identical
frozen convolutions with torch CUDA.  Inner OOF selects V69 noop or fixed
.25/.50 blends; outer labels are evaluation-only and every material failure
restores the exact entire atomic V69 frame with confidence/high_conf unchanged.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_name] = "2"

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v188_multichannel_causal_persistent_homology_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V191" / "LOCAL_FORBIDDEN"
VERSION = 191
HYPOTHESIS = "MULTICHANNEL_CAUSAL_FIXED_RANDOM_CONVOLUTIONAL_SCATTERING_RESERVOIR_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v188_multichannel_causal_persistent_homology_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "14d351120e6b62425d1788e8efe56007b766b28fa7be56ed6f6a1172d6b4f194"

CHANNEL_N = 8
HORIZON_N = 8
RESERVOIR_CHANNELS = 12
SUMMARY_N = 6
RESERVOIR_FEATURE_DIMENSION = 2 * RESERVOIR_CHANNELS * SUMMARY_N + 4 * CHANNEL_N
CHANNEL_CLIP = 7.0
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 19101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "FIXED_RANDOM_CONV_SCATTERING_W0.25", "weight": 0.25},
    {"name": "FIXED_RANDOM_CONV_SCATTERING_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
metric = scaffold.metric
controller = scaffold.controller
numeric = scaffold.numeric
observed_past_to_recent_path = scaffold.observed_past_to_recent_path
equal_event_group_path = scaffold.equal_event_group_path
TrainChannelScale = scaffold.TrainChannelScale
blend_probability = scaffold.blend_probability


class RobustFeatureScale:
    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == RESERVOIR_FEATURE_DIMENSION, "V191 feature shape invalid")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(result.shape[1] == RESERVOIR_FEATURE_DIMENSION and np.isfinite(result).all(), "V191 scaled feature invalid")
        return result


def _fixed_bank(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bank = rng.standard_normal(shape).astype(np.float64)
    norm = np.sqrt(np.sum(bank * bank, axis=tuple(range(1, bank.ndim)), keepdims=True))
    return bank / np.maximum(norm, 1e-12)


BANK1 = _fixed_bank((RESERVOIR_CHANNELS, 1, 3, 3), SEED + 1)
BANK2 = _fixed_bank((RESERVOIR_CHANNELS, RESERVOIR_CHANNELS, 3, 3), SEED + 2)
BANK_SHA256 = array_sha256(np.concatenate([BANK1.ravel(), BANK2.ravel()]))

STATIC_SPEC = {
    "path_shape": [HORIZON_N, CHANNEL_N],
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent by eight immutable semantic channels",
    "reservoir": "two frozen seeded 3x3 random convolution banks with reflect padding and absolute-tanh scattering nonlinearity",
    "reservoir_channels": RESERVOIR_CHANNELS,
    "filter_bank_sha256": BANK_SHA256,
    "pooling": "per-map global mean/std/max plus old-half, recent-half and recent-minus-old means; four raw summaries per semantic channel",
    "feature_dimension": RESERVOIR_FEATURE_DIMENSION,
    "head": "one equal-event-group/class-balanced C=0.5 ridge logistic",
    "full_backend": "torch CUDA0 identical frozen convolutions",
    "bounded_backend": "NumPy CPU",
    "trained_filter_mask_decoder_or_target_batch_fit": False,
}
require(RESERVOIR_FEATURE_DIMENSION == 176, "V191 reservoir feature dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V188 scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "V36/V69 authority, chronology, observed path, metrics, gate, bootstrap and atomic IO only; no V188 output is read"}
    audit["v191_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


def _conv_numpy(tensor: np.ndarray, bank: np.ndarray) -> np.ndarray:
    padded = np.pad(tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(-2, -1))
    return np.einsum("nihwkl,oikl->nohw", windows, bank, optimize=True)


def _reservoir_layers_numpy(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    layer1 = np.abs(np.tanh(_conv_numpy(path[:, None, :, :], BANK1)))
    layer2 = np.abs(np.tanh(_conv_numpy(layer1, BANK2)))
    return layer1, layer2, "numpy_cpu"


def _reservoir_layers_cuda(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "V191 full runtime did not bind GPU0")
    import torch
    import torch.nn.functional as functional
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "V191 CUDA GPU0 unavailable")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    with torch.inference_mode():
        tensor = torch.from_numpy(path[:, None, :, :].astype(np.float32)).to(device)
        weight1 = torch.from_numpy(BANK1.astype(np.float32)).to(device)
        weight2 = torch.from_numpy(BANK2.astype(np.float32)).to(device)
        layer1 = torch.abs(torch.tanh(functional.conv2d(functional.pad(tensor, (1, 1, 1, 1), mode="reflect"), weight1)))
        layer2 = torch.abs(torch.tanh(functional.conv2d(functional.pad(layer1, (1, 1, 1, 1), mode="reflect"), weight2)))
        first = layer1.cpu().numpy().astype(np.float64); second = layer2.cpu().numpy().astype(np.float64)
    return first, second, "torch_cuda0"


def reservoir_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V191 path shape invalid")
    use_cuda = bool(CONTROLLER_OUTPUT and os.environ.get("CUDA_VISIBLE_DEVICES") == "0")
    layer1, layer2, backend = _reservoir_layers_cuda(path) if use_cuda else _reservoir_layers_numpy(path)
    blocks: list[np.ndarray] = []
    for layer in (layer1, layer2):
        blocks.extend([
            layer.mean(axis=(2, 3)), layer.std(axis=(2, 3)), layer.max(axis=(2, 3)),
            layer[:, :, :4, :].mean(axis=(2, 3)), layer[:, :, 4:, :].mean(axis=(2, 3)),
            layer[:, :, 4:, :].mean(axis=(2, 3)) - layer[:, :, :4, :].mean(axis=(2, 3)),
        ])
    raw = [path.mean(axis=1), path.std(axis=1), path[:, -1, :] - path[:, 0, :], np.mean(np.abs(np.diff(path, axis=1)), axis=1)]
    result = np.concatenate(blocks + raw, axis=1)
    require(result.shape == (len(path), RESERVOIR_FEATURE_DIMENSION) and np.isfinite(result).all(), "V191 feature invalid")
    return result, {"rows": len(path), "feature_dimension": result.shape[1], "feature_sha256": array_sha256(result), "backend": backend, "filter_bank_sha256": BANK_SHA256, "frozen_random_filters": True, "two_3x3_convolution_layers": True, "absolute_tanh_scattering": True, "trainable_encoder_decoder_or_masked_reconstruction": False, "target_batch_fit": False, "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": True, "v82_masked_tabular_autoencoder": False, "v95_haar_wavelet_scattering": False, "v55_trained_temporal_cnn": False, "v150_mlp_mixer": False, "v153_axial_attention": False}


def reservoir_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V191 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path); group_path = channel_scale.transform(group_path); target_path = channel_scale.transform(target_path)
    train_feature, train_reservoir = reservoir_features(group_path); target_feature, target_reservoir = reservoir_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature); train_design = feature_scale.transform(train_feature); target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float); require(np.all(class_n > 0), "V191 train cut missing class")
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, group_target, sample_weight=sample_weight); probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all() and int(head.n_iter_[0]) < 1000, "V191 head invalid")
    return probability, {"train_n":len(ordered),"train_event_group_n":len(group_target),"target_n":len(target_frame),"causal_numeric_only":True,"categorical_text_source_ticker_or_issuer_feature_n":0,"train_transform":train_transform,"target_transform":target_transform,"grouping":grouping,"static_spec":STATIC_SPEC,"train_path":train_path_audit,"target_path":target_path_audit,"train_geometry":train_reservoir,"target_geometry":target_reservoir,"channel_scaling":{"train_only":True,"median_sha256":array_sha256(channel_scale.median),"scale_sha256":array_sha256(channel_scale.scale),"clip":CHANNEL_CLIP},"feature_scaling":{"train_only":True,"median_sha256":array_sha256(feature_scale.median),"scale_sha256":array_sha256(feature_scale.scale),"clip":FEATURE_CLIP},"head":{"type":"fixed equal-event-group/class-balanced ridge logistic","C":RIDGE_LOGISTIC_C,"solver":"liblinear","iterations":int(head.n_iter_[0]),"class_event_group_n":class_n.astype(int).tolist(),"coefficient_l2":float(np.linalg.norm(head.coef_)),"coefficient_sha256":array_sha256(head.coef_),"intercept":head.intercept_.tolist()},"target_rows_used_for_channel_or_feature_scale_or_head":False,"target_rows_used_for_robust_scale_geometry_scale_or_head":False,"target_labels_used":False,"filters_trained_with_train_or_target_rows":False,"haar_wavelet_or_scattering":False,"rough_path_signature_or_levy_area":False,"increment_spd_or_covariance":False,"fourier_cross_spectrum_or_dmd":False,"cusum_changepoint_or_recurrence":False,"natural_visibility_graph_or_ordinal_motif":False,"source_time_environment_transport_or_projection":False,"prediction":{"mean":float(probability.mean()),"std":float(probability.std()),"probability_sha256":array_sha256(probability)}}


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION=VERSION; base.HYPOTHESIS=HYPOTHESIS; base.STATIC_SPEC=STATIC_SPEC; base.FRENET_FEATURE_DIMENSION=RESERVOIR_FEATURE_DIMENSION; base.ARCHITECTURES=ARCHITECTURES; base.SEED=SEED; base.frenet_direction=reservoir_direction


def choose_inner(inner_train:pd.DataFrame,inner_valid:pd.DataFrame,inner_champion:pd.DataFrame)->tuple[dict[str,Any],dict[str,Any]]:
    challenger,audit=reservoir_direction(inner_train,inner_valid);baseline=inner_champion.prob.to_numpy(float);confidence=inner_champion.confidence_signal.to_numpy(float);high=bool_series(inner_champion.high_conf).to_numpy(bool);baseline_metric=metric(inner_valid,baseline,confidence,high)
    trials=[{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.0,"balanced_accuracy_delta":0.0,"all_trade_net_delta":0.0,"worst_supported_source_ba_delta":0.0,"supported_source_ba_delta":{},"eligible":True,"score":2.0*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}]
    geometry=audit["train_geometry"];model_eligible=bool(audit["causal_numeric_only"] and audit["grouping"]["equal_event_group_weighting"] and audit["channel_scaling"]["train_only"] and audit["feature_scaling"]["train_only"] and geometry["frozen_random_filters"] and geometry["two_3x3_convolution_layers"] and geometry["absolute_tanh_scattering"] and not geometry["trainable_encoder_decoder_or_masked_reconstruction"] and not geometry["target_batch_fit"] and not audit["filters_trained_with_train_or_target_rows"] and not audit["target_rows_used_for_channel_or_feature_scale_or_head"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]);current=metric(inner_valid,probability,confidence,high);auc_delta=current["auc"]-baseline_metric["auc"];ba_delta=current["balanced_accuracy"]-baseline_metric["balanced_accuracy"];net_delta=current["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"];source_floor,source_deltas=base.supported_source_ba_delta(inner_valid,baseline,probability);trials.append({"name":architecture["name"],"architecture":architecture,"metrics":current,"auc_delta":auc_delta,"balanced_accuracy_delta":ba_delta,"all_trade_net_delta":net_delta,"worst_supported_source_ba_delta":source_floor,"supported_source_ba_delta":source_deltas,"model_eligible":model_eligible,"eligible":bool(model_eligible and ba_delta>=-0.005 and net_delta>=-0.0005 and source_floor>=-0.010),"score":2.0*current["auc"]+current["balanced_accuracy"]})
    selected=max([x for x in trials if x["eligible"]],key=lambda x:(x["score"],x["auc_delta"],x["all_trade_net_delta"],x["name"]=="V69_NOOP"));return {"selection_rule":"inner-past OOF only; exact V69 noop or fixed .25/.50 frozen random convolutional scattering blend","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"filter_count_seed_summary_head_blend_or_safety_micro_tuning":False},audit


def run_nested(dev:pd.DataFrame,champion:pd.DataFrame,smoke:bool,smoke_market:str="US")->tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    configure_base();base.choose_inner=choose_inner
    with redirect_stdout(StringIO()):diagnostic,evidence,audits=base.run_nested(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v146_model":"v191_model"});evidence=evidence.rename(columns={"frenet_prob":"random_conv_scattering_prob","v146_model":"v191_model"})
    for audit in audits:
        selected=audit["policy"]["selected"];print(f"[V191 RANDOM CONV] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}",flush=True)
    return diagnostic,evidence,audits


def paired_nested_bootstrap(evidence:pd.DataFrame,draws:int)->dict[str,Any]:
    configure_base();result=_BASE_BOOTSTRAP(evidence,draws);result["method"]="paired market-fold-time block bootstrap over inner-locked V191 frozen random convolutional scattering policy";result["seed"]=SEED;result["standardized"]=True;return result


def evaluate(champion:pd.DataFrame,diagnostic:pd.DataFrame,evidence:pd.DataFrame,audits:list[dict[str,Any]],draws:int,smoke:bool)->dict[str,Any]:
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;result=base.evaluate(champion,diagnostic,evidence,audits,draws,smoke);checks=result["material_checks"];checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract=bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["feature_dimension"]==RESERVOIR_FEATURE_DIMENSION and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["train_geometry"]["frozen_random_filters"] and audit[side]["train_geometry"]["two_3x3_convolution_layers"] and audit[side]["train_geometry"]["absolute_tanh_scattering"] and not audit[side]["train_geometry"]["trainable_encoder_decoder_or_masked_reconstruction"] and not audit[side]["train_geometry"]["target_batch_fit"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["filters_trained_with_train_or_target_rows"] and not audit[side]["target_rows_used_for_channel_or_feature_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit","outer_model_audit")))
    checks["multichannel_causal_fixed_random_convolutional_scattering_reservoir_contract_verified"]=contract;result["material_pass"]=bool(not smoke and all(checks.values()))
    if not result["material_pass"]:result["selected_frame"]=champion.copy();result["selected_summary"]=result["baseline_summary"];result["fallback"]={"activated":True,"policy":"exact entire V69 DataFrame","exact_entire_frame_verified":bool(result["selected_frame"].equals(champion))};require(result["selected_frame"].equals(champion),"V191 fallback not exact V69")
    return result


def material_gate(evaluation:dict[str,Any])->dict[str,Any]:
    checks=evaluation["material_checks"];gate={"contract":HYPOTHESIS,"checks":checks,"passed":int(sum(bool(x) for x in checks.values())),"total":len(checks),"material_pass":bool(evaluation["material_pass"]),"nested_bootstrap":evaluation["nested_bootstrap"],"native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"],"current_hypothesis_not_inherited_from_v69":True,"fallback_policy":"candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"};require(gate["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V191 nested key mismatch");return gate


def enforce_output_conflict_fail_closed(out:Path)->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT),"V191 output requires controller authority");require(out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V191 output target mismatch");require(out.resolve().parent==(ROOT/"research"/"staging"/"V191").resolve(),"V191 output outside staging/V191");require(out.exists()and out.is_dir(),"V191 staging child must preexist");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing),f"V191 prefiles missing: {sorted(required-existing)}");require(not(existing-allowed),f"V191 unexpected conflict: {sorted(existing-allowed)}");return {"controller_bound":True,"target_preexisting":True,"required_controller_prefiles":sorted(required),"observed_controller_prefiles":sorted(existing),"unexpected_prefiles":[],"conflict_policy":"allow exact controller prefiles only; fail closed before write"}


def write_outputs(out:Path,authority:dict[str,Any],evidence:pd.DataFrame,audits:list[dict[str,Any]],evaluation:dict[str,Any])->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V191 output not controller-bound");conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in evaluation.items()if not k.endswith("_frame")};gate=material_gate(evaluation);selected=json.loads(json.dumps(clean(evaluation["selected_summary"])));selected["material_gate"]=gate;require("bootstrap" in selected["robustness"],"V191 native bootstrap missing")
    reports={"MODEL_COMPARISON.json":{"version":"V191","hypothesis":HYPOTHESIS,"status":evaluation["status"],"models":{"V69_CHAMPION":evaluation["baseline_summary"],"V191_DIAGNOSTIC_RANDOM_CONV_SCATTERING":evaluation["candidate_summary"],"V191_FAIL_CLOSED_SELECTED":selected},"evaluation":compact,"authority_audit":authority,"nested_fold_audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V191","hypothesis":HYPOTHESIS,"status":evaluation["status"],"opened_dev_only":True,"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"material_checks":evaluation["material_checks"],"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"SOURCE_TRANSFER_REPORT.json":{"version":"V191","hypothesis":HYPOTHESIS,"status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate,"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"V191_RANDOM_CONV_SCATTERING_REPORT.json":{"version":"V191","hypothesis":HYPOTHESIS,"static_spec":STATIC_SPEC,"architectures":ARCHITECTURES,"nested_fold_audits":audits,"evaluation":compact,"authority_audit":authority},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"version":"V191","contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"version":"V191","controller_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"version":"V191","status":"MATCH","canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected),"material_gate":gate},"RUN_STATUS.json":{"version":VERSION,"hypothesis":HYPOTHESIS,"status":evaluation["status"],"material_pass":evaluation["material_pass"],"champion_changed":evaluation["material_pass"],"seal_state":"UNOPENED","seal_authorized":False,"output_conflict_audit":conflict,"completed_at":pd.Timestamp.now(tz="UTC").isoformat()}}
    for name,payload in reports.items():atomic_json(payload,out/name)
    atomic_csv(evidence,out/"V191_RANDOM_CONV_SCATTERING_OUTER_EVIDENCE.csv.gz");atomic_csv(evaluation["selected_frame"],out/"V191_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in sorted(out.iterdir())if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"hypothesis":HYPOTHESIS,"authority_experiment_id":EXPECTED_V69_EXPERIMENT_ID,"files":files},out/"ARTIFACT_MANIFEST.json");return {"output":str(out),"artifact_count":len(files)+1,"manifest_sha256":sha256(out/"ARTIFACT_MANIFEST.json")}


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));modes.add_argument("--audit-only",action="store_true");modes.add_argument("--smoke-test",action="store_true");modes.add_argument("--support-probe",action="store_true");modes.add_argument("--full-run",action="store_true");parser.add_argument("--smoke-market",choices=("US","KR"),default="US");parser.add_argument("--output",type=Path,default=None);args=parser.parse_args()
    if CONTROLLER_OUTPUT:require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None,"explicit mode/output conflicts with controller-bound no-argument V191 full run");args.full_run=True;args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None:args.output=DEFAULT_OUT
    if args.full_run:require(bool(CONTROLLER_OUTPUT),"V191 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT");require(args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V191 controller full-run output mismatch")
    return args


def main()->None:
    args=parse_args();bounded=bool(args.audit_only or args.smoke_test or args.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else {"mode":"controller_bound_full","gpu_model_calls":"torch CUDA0 frozen convolution only","numeric_thread_cap":2};authority=verify_authority();dev,champion=load_authorized()
    if args.full_run:require(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","V191 controller full requires runtime_limits CUDA0")
    if args.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"dev_rows":len(dev),"v69_rows":len(champion),"static_spec":STATIC_SPEC,"architecture":"frozen seeded two-layer random convolutional scattering reservoir; balanced ridge","preferred_autoencoder_rejected_due_v82_collision":True,"canonical_controller_gate_checks":14,"controller_contract":{"no_argument_market_bio_version_output_full":True,"controller_output_exact_binding":True,"explicit_mode_or_output_conflict_fails_closed":True,"direct_full_without_controller_fails_closed":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"nested_bootstrap_key":"selected.material_gate.nested_bootstrap","native_robustness_bootstrap_preserved":True,"source_transfer_current_gate":True,"exact_entire_v69_fallback":True},"output_written":False}),ensure_ascii=False,indent=2));return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,smoke=bounded,smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market=="KR","V191 support reserved for KR:2");audit=audits[0];non=[x for x in audit["policy"]["trials"]if x["architecture"]is not None];best=max(non,key=lambda x:(x["eligible"],x["score"],x["auc_delta"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","hypothesis":HYPOTHESIS,"runtime":runtime,"market":"KR","fold":2,"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audit["inner_model_audit"],"outer_model_audit":audit["outer_model_audit"],"selection_locked_before_outer_evaluation":True,"exact_entire_v69_fallback_contract":True,"output_written":False}),ensure_ascii=False,indent=2));return
        evaluation=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,smoke=args.smoke_test)
    non=[x for x in audits[0]["policy"]["trials"]if x["architecture"]is not None];best=max(non,key=lambda x:(x["eligible"],x["score"],x["auc_delta"]));summary={"status":"SMOKE_OK"if args.smoke_test else evaluation["status"],"hypothesis":HYPOTHESIS,"runtime":runtime,"folds_executed":len(audits),"smoke_market":args.smoke_market if args.smoke_test else None,"selected_models":[a["policy"]["selected"]["name"]for a in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"canonical_gate":evaluation["selected_summary"]["research_gate"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test:summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audits[0]["inner_model_audit"],"outer_model_audit":audits[0]["outer_model_audit"],"candidate_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]});print(json.dumps(clean(summary),ensure_ascii=False,indent=2));return
    result=write_outputs(args.output,authority,evidence,audits,evaluation);print(json.dumps(clean({**summary,"output_written":True,"controller":result}),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
