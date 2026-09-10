"""V174 multichannel causal morphological-granulometry direction.

Every immutable V36 eight-channel by eight-horizon causal path is scaled from
the strict-past training cut.  Fixed flat grayscale erosions and dilations form
openings and closings at radii one, two and three.  Per-channel granulometric
top-/bottom-hat area and energy, pattern-spectrum increments, and a fixed
alternating sequential-filter residual summarize peak/valley size structure.
One equal-event-group/class-balanced fixed ridge-logit head learns direction.
There is no learned structuring element, target-batch fit or scale search.

This is not V173's weighted grid-Laplacian eigenspectrum, V171's wavelet-leader
scaling, V172's Hankel SSA, or V175's Burg lattice recursion.  Inner OOF selects
exact V69 or fixed .25/.50 blends; outer labels are evaluation-only after the
35-minute embargo/event purge.  V69 confidence/high_conf remain exact and any
material failure restores the exact entire atomic V69 frame.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_thread_variable] = "2"

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

import experiment_v171_multichannel_causal_wavelet_leader_multifractal_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V174" / "LOCAL_FORBIDDEN"
VERSION = 174
HYPOTHESIS = "MULTICHANNEL_CAUSAL_MORPHOLOGICAL_GRANULOMETRY_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v171_multichannel_causal_wavelet_leader_multifractal_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "9e2dd8f0f61b7bb1cea7f8406d11dec28744a644725f38fbaaf83a9d6a921989"

CHANNEL_N = 8
HORIZON_N = 8
RADII = (1, 2, 3)
FEATURES_PER_RADIUS = 6
PATTERN_SPECTRUM_FEATURES = 2 * len(RADII)
ASF_FEATURES = 6
FEATURES_PER_CHANNEL = len(RADII) * FEATURES_PER_RADIUS + PATTERN_SPECTRUM_FEATURES + ASF_FEATURES
MORPH_FEATURE_DIMENSION = CHANNEL_N * FEATURES_PER_CHANNEL
CHANNEL_CLIP = 7.0
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 17401
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "MORPHOLOGICAL_GRANULOMETRY_W0.25", "weight": 0.25},
    {"name": "MORPHOLOGICAL_GRANULOMETRY_W0.50", "weight": 0.50},
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
blend_probability = base.blend_probability

STATIC_SPEC = {
    "path_shape": [HORIZON_N, CHANNEL_N],
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "structuring_element": "fixed flat one-dimensional windows with edge replication",
    "radii": list(RADII),
    "operators": ["erosion", "dilation", "opening", "closing"],
    "granulometry": ["top-hat area/RMS", "bottom-hat area/RMS", "opening/closing total variation", "successive area pattern spectra"],
    "alternating_sequential_filter": "opening then closing successively at radii 1,2,3",
    "feature_dimension": MORPH_FEATURE_DIMENSION,
    "head": "one equal-event-group/class-balanced C=0.5 ridge logistic",
    "learned_structuring_element_or_radius_search": False,
    "target_batch_fit": False,
}
require(FEATURES_PER_CHANNEL == 30 and MORPH_FEATURE_DIMENSION == 240, "V174 morphology feature dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V171 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, observed path, metrics, gate, bootstrap and atomic IO only; no V171 output is read"}
    audit["v174_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.zeros(MORPH_FEATURE_DIMENSION)
        self.scale = np.ones(MORPH_FEATURE_DIMENSION)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.shape[1] == MORPH_FEATURE_DIMENSION, "V174 feature shape invalid")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V174 scaled feature invalid")
        return result


def flat_filter(series: np.ndarray, radius: int, maximum: bool) -> np.ndarray:
    padded = np.pad(np.asarray(series, float), radius, mode="edge")
    result = np.empty_like(series, dtype=float)
    for index in range(len(series)):
        window = padded[index:index + 2 * radius + 1]
        result[index] = np.max(window) if maximum else np.min(window)
    return result


def erosion(series: np.ndarray, radius: int) -> np.ndarray:
    return flat_filter(series, radius, maximum=False)


def dilation(series: np.ndarray, radius: int) -> np.ndarray:
    return flat_filter(series, radius, maximum=True)


def opening(series: np.ndarray, radius: int) -> np.ndarray:
    return dilation(erosion(series, radius), radius)


def closing(series: np.ndarray, radius: int) -> np.ndarray:
    return erosion(dilation(series, radius), radius)


def channel_granulometry(series: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(series.shape == (HORIZON_N,), "V174 channel path shape invalid")
    features: list[float] = []
    top_areas: list[float] = []
    bottom_areas: list[float] = []
    opening_violations = 0
    closing_violations = 0
    for radius in RADII:
        opened = opening(series, radius)
        closed = closing(series, radius)
        opening_violations += int(np.sum(opened > series + 1e-10))
        closing_violations += int(np.sum(closed < series - 1e-10))
        top = np.maximum(series - opened, 0.0)
        bottom = np.maximum(closed - series, 0.0)
        top_area = float(np.mean(top)); bottom_area = float(np.mean(bottom))
        top_areas.append(top_area); bottom_areas.append(bottom_area)
        features.extend([top_area, float(np.sqrt(np.mean(np.square(top)))), bottom_area, float(np.sqrt(np.mean(np.square(bottom)))), float(np.mean(np.abs(np.diff(opened)))), float(np.mean(np.abs(np.diff(closed))))])
    features.extend(np.diff(np.asarray([0.0] + top_areas)).tolist())
    features.extend(np.diff(np.asarray([0.0] + bottom_areas)).tolist())
    filtered = np.asarray(series, float)
    for radius in RADII:
        filtered = closing(opening(filtered, radius), radius)
    residual = series - filtered
    features.extend([float(np.mean(residual)), float(np.std(residual)), float(np.mean(np.abs(residual))), float(np.sqrt(np.mean(np.square(residual)))), float(np.mean(np.abs(np.diff(filtered)))), float(filtered[-1] - filtered[0])])
    result = np.asarray(features, dtype=float)
    require(result.shape == (FEATURES_PER_CHANNEL,) and np.isfinite(result).all(), "V174 channel morphology invalid")
    return result, {"opening_anti_extensive_violations": opening_violations, "closing_extensive_violations": closing_violations, "top_area_nondecreasing": bool(np.all(np.diff(top_areas) >= -1e-10)), "bottom_area_nondecreasing": bool(np.all(np.diff(bottom_areas) >= -1e-10))}


def morphology_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V174 path shape invalid")
    result = np.empty((len(path), MORPH_FEATURE_DIMENSION), dtype=float)
    violations = 0
    monotone = 0
    for row in range(len(path)):
        parts = []
        for channel in range(CHANNEL_N):
            part, audit = channel_granulometry(path[row, :, channel]); parts.append(part)
            violations += audit["opening_anti_extensive_violations"] + audit["closing_extensive_violations"]
            monotone += int(audit["top_area_nondecreasing"] and audit["bottom_area_nondecreasing"])
        result[row] = np.concatenate(parts)
    require(np.isfinite(result).all() and float(np.std(result)) > 1e-10, "V174 morphology feature degenerate")
    return result, {"rows": len(result), "feature_dimension": result.shape[1], "feature_sha256": array_sha256(result), "radii": list(RADII), "flat_erosion_dilation_opening_closing": True, "granulometric_pattern_spectrum": True, "top_bottom_hat_energy": True, "alternating_sequential_filter": True, "operator_order_violation_n": violations, "channel_row_monotone_pattern_fraction": float(monotone / max(len(path) * CHANNEL_N, 1)), "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": True, "learned_structuring_element_or_radius": False, "target_batch_fit": False, "grid_laplacian_or_eigenspectrum": False, "wavelet_leader_or_scattering": False, "hankel_ssa_or_burg_lattice": False}


def morphology_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V174 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered); train_state, train_transform = processor.transform(ordered); target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)]); target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path); group_path = channel_scale.transform(group_path); target_path = channel_scale.transform(target_path)
    train_feature, train_morphology = morphology_features(group_path); target_feature, target_morphology = morphology_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature); train_design = feature_scale.transform(train_feature); target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float); sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED); head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6); require(np.isfinite(probability).all() and int(head.n_iter_[0]) < 1000, "V174 head invalid")
    return probability, {"train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit, "train_geometry": train_morphology, "target_geometry": target_morphology, "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": CHANNEL_CLIP}, "feature_scaling": {"train_only": True, "median_sha256": array_sha256(feature_scale.median), "scale_sha256": array_sha256(feature_scale.scale), "clip": FEATURE_CLIP}, "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "class_event_group_n": class_n.astype(int).tolist(), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "intercept": head.intercept_.tolist()}, "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False, "v173_grid_laplacian": False, "v171_wavelet_leader": False, "v172_hankel_ssa": False, "v175_burg_lattice": False, "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)}}


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION; base.HYPOTHESIS = HYPOTHESIS; base.STATIC_SPEC = STATIC_SPEC; base.FRENET_FEATURE_DIMENSION = MORPH_FEATURE_DIMENSION; base.ARCHITECTURES = ARCHITECTURES; base.SEED = SEED; base.frenet_direction = morphology_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = morphology_direction(inner_train, inner_valid); baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.0,"balanced_accuracy_delta":0.0,"all_trade_net_delta":0.0,"worst_supported_source_ba_delta":0.0,"supported_source_ba_delta":{},"eligible":True,"score":2.0*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["channel_scaling"]["train_only"] and model_audit["feature_scaling"]["train_only"] and model_audit["train_geometry"]["flat_erosion_dilation_opening_closing"] and model_audit["train_geometry"]["granulometric_pattern_spectrum"] and model_audit["train_geometry"]["top_bottom_hat_energy"] and model_audit["train_geometry"]["alternating_sequential_filter"] and not model_audit["train_geometry"]["learned_structuring_element_or_radius"] and not model_audit["train_geometry"]["target_batch_fit"] and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not model_audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]);current=metric(inner_valid,probability,confidence,high);auc_delta=current["auc"]-baseline_metric["auc"];ba_delta=current["balanced_accuracy"]-baseline_metric["balanced_accuracy"];net_delta=current["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"];source_floor,source_deltas=base.supported_source_ba_delta(inner_valid,baseline,probability);trials.append({"name":architecture["name"],"architecture":architecture,"metrics":current,"auc_delta":auc_delta,"balanced_accuracy_delta":ba_delta,"all_trade_net_delta":net_delta,"worst_supported_source_ba_delta":source_floor,"supported_source_ba_delta":source_deltas,"model_eligible":model_eligible,"eligible":bool(model_eligible and ba_delta>=-0.005 and net_delta>=-0.0005 and source_floor>=-0.010),"score":2.0*current["auc"]+current["balanced_accuracy"]})
    selected=max([t for t in trials if t["eligible"]],key=lambda t:(t["score"],t["auc_delta"],t["all_trade_net_delta"],t["name"]=="V69_NOOP"));return {"selection_rule":"inner-past OOF only; exact V69 noop or fixed .25/.50 morphological granulometry blend with fixed safety","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"radii_operator_summary_head_blend_or_source_floor_micro_tuning":False},model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    configure_base();base.choose_inner=choose_inner
    with redirect_stdout(StringIO()):diagnostic,evidence,audits=base.run_nested(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v146_model":"v174_model"});evidence=evidence.rename(columns={"frenet_prob":"morphology_prob","v146_model":"v174_model"})
    for audit in audits:
        selected=audit["policy"]["selected"];print(f"[V174 MORPH GRANULOMETRY] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}",flush=True)
    return diagnostic,evidence,audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str,Any]:
    configure_base();result=_BASE_BOOTSTRAP(evidence,draws);result["method"]="paired market-fold-time block bootstrap over inner-locked V174 morphology policy";result["seed"]=SEED;result["standardized"]=True;return result


def evaluate(champion: pd.DataFrame,diagnostic:pd.DataFrame,evidence:pd.DataFrame,audits:list[dict[str,Any]],draws:int,smoke:bool)->dict[str,Any]:
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;result=base.evaluate(champion,diagnostic,evidence,audits,draws,smoke);checks=result["material_checks"];checks.pop("multichannel_discrete_frenet_curvature_contract_verified");contract=bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["feature_dimension"]==MORPH_FEATURE_DIMENSION and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["train_geometry"]["flat_erosion_dilation_opening_closing"] and audit[side]["train_geometry"]["granulometric_pattern_spectrum"] and audit[side]["train_geometry"]["top_bottom_hat_energy"] and audit[side]["train_geometry"]["alternating_sequential_filter"] and not audit[side]["train_geometry"]["grid_laplacian_or_eigenspectrum"] and not audit[side]["train_geometry"]["wavelet_leader_or_scattering"] and not audit[side]["train_geometry"]["hankel_ssa_or_burg_lattice"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit","outer_model_audit")));checks["multichannel_causal_morphological_granulometry_contract_verified"]=contract;result["material_pass"]=bool(not smoke and all(checks.values()))
    if not result["material_pass"]:result["selected_frame"]=champion.copy();result["selected_summary"]=result["baseline_summary"];result["fallback"]={"activated":True,"policy":"exact entire V69 DataFrame","exact_entire_frame_verified":bool(result["selected_frame"].equals(champion))};require(result["selected_frame"].equals(champion),"V174 fallback not exact V69")
    return result


def material_gate(evaluation:dict[str,Any])->dict[str,Any]:
    checks=evaluation["material_checks"];gate={"contract":HYPOTHESIS,"checks":checks,"passed":int(sum(bool(v) for v in checks.values())),"total":len(checks),"material_pass":bool(evaluation["material_pass"]),"nested_bootstrap":evaluation["nested_bootstrap"],"native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"],"current_hypothesis_not_inherited_from_v69":True,"fallback_policy":"candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"};require(gate["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V174 nested key mismatch");return gate


def enforce_output_conflict_fail_closed(out:Path)->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT),"V174 output requires controller authority");require(out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V174 output target mismatch");require(out.resolve().parent==(ROOT/"research"/"staging"/"V174").resolve(),"V174 output outside staging/V174");require(out.exists()and out.is_dir(),"V174 staging child must preexist");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={p.name for p in out.iterdir()};require(required.issubset(existing),f"V174 prefiles missing: {sorted(required-existing)}");require(not(existing-allowed),f"V174 unexpected conflict: {sorted(existing-allowed)}");return {"controller_bound":True,"target_preexisting":True,"required_controller_prefiles":sorted(required),"observed_controller_prefiles":sorted(existing),"unexpected_prefiles":[],"conflict_policy":"allow exact controller prefiles only; fail closed before write"}


def write_outputs(out:Path,authority:dict[str,Any],evidence:pd.DataFrame,audits:list[dict[str,Any]],evaluation:dict[str,Any])->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V174 output not controller-bound");conflict=enforce_output_conflict_fail_closed(out);compact={k:v for k,v in evaluation.items()if not k.endswith("_frame")};gate=material_gate(evaluation);selected=json.loads(json.dumps(clean(evaluation["selected_summary"])));selected["material_gate"]=gate;require("bootstrap"in selected["robustness"],"V174 native bootstrap missing");reports={"MODEL_COMPARISON.json":{"version":"V174","hypothesis":HYPOTHESIS,"status":evaluation["status"],"models":{"V69_CHAMPION":evaluation["baseline_summary"],"V174_DIAGNOSTIC_MORPHOLOGY":evaluation["candidate_summary"],"V174_FAIL_CLOSED_SELECTED":selected},"evaluation":compact,"authority_audit":authority,"nested_fold_audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V174","hypothesis":HYPOTHESIS,"status":evaluation["status"],"opened_dev_only":True,"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"material_checks":evaluation["material_checks"],"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"SOURCE_TRANSFER_REPORT.json":{"version":"V174","hypothesis":HYPOTHESIS,"status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate,"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"V174_MORPHOLOGICAL_GRANULOMETRY_REPORT.json":{"version":"V174","hypothesis":HYPOTHESIS,"static_spec":STATIC_SPEC,"architectures":ARCHITECTURES,"nested_fold_audits":audits,"evaluation":compact,"authority_audit":authority},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"version":"V174","hypothesis":HYPOTHESIS,"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"version":"V174","hypothesis":HYPOTHESIS,"controller_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"version":"V174","status":"MATCH","canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected),"material_gate":gate},"RUN_STATUS.json":{"version":VERSION,"hypothesis":HYPOTHESIS,"status":evaluation["status"],"material_pass":evaluation["material_pass"],"champion_changed":evaluation["material_pass"],"seal_state":"UNOPENED","seal_authorized":False,"output_conflict_audit":conflict,"completed_at":pd.Timestamp.now(tz="UTC").isoformat()}}
    for name,payload in reports.items():atomic_json(payload,out/name)
    atomic_csv(evidence,out/"V174_MORPHOLOGY_OUTER_EVIDENCE.csv.gz");atomic_csv(evaluation["selected_frame"],out/"V174_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}for p in sorted(out.iterdir())if p.is_file()and p.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"hypothesis":HYPOTHESIS,"authority_experiment_id":EXPECTED_V69_EXPERIMENT_ID,"files":files},out/"ARTIFACT_MANIFEST.json");return {"output":str(out),"artifact_count":len(files)+1,"manifest_sha256":sha256(out/"ARTIFACT_MANIFEST.json")}


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));modes.add_argument("--audit-only",action="store_true");modes.add_argument("--smoke-test",action="store_true");modes.add_argument("--support-probe",action="store_true");modes.add_argument("--full-run",action="store_true");parser.add_argument("--smoke-market",choices=("US","KR"),default="US");parser.add_argument("--output",type=Path,default=None);args=parser.parse_args()
    if CONTROLLER_OUTPUT:require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None,"explicit mode/output conflicts with controller-bound no-argument V174 full run");args.full_run=True;args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None:args.output=DEFAULT_OUT
    if args.full_run:require(bool(CONTROLLER_OUTPUT),"V174 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT");require(args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V174 controller full-run output mismatch")
    return args


def main()->None:
    args=parse_args();bounded=bool(args.audit_only or args.smoke_test or args.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller_bound_full","thread_policy":"authorized parent inherited","gpu_model_calls":0};authority=verify_authority();dev,champion=load_authorized()
    if args.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"dev_rows":len(dev),"v69_rows":len(champion),"features":FEATURES,"static_spec":STATIC_SPEC,"architecture":"fixed flat min/max morphology radii1/2/3; granulometry/top-bottom-hat/ASF features; balanced ridge","causal_numeric_only":True,"target_labels_used":False,"canonical_controller_gate_checks":14,"controller_contract":{"no_argument_market_bio_version_output_full":True,"controller_output_exact_binding":True,"explicit_mode_or_output_conflict_fails_closed":True,"direct_full_without_controller_fails_closed":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"current_material_gate":True,"nested_bootstrap_key":"selected.material_gate.nested_bootstrap","standardized_nested_bootstrap":True,"native_robustness_bootstrap_preserved":True,"source_transfer_current_gate":True,"exact_entire_v69_fallback":True},"output_written":False}),ensure_ascii=False,indent=2));return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,smoke=bounded,smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market=="KR","V174 support reserved for KR:2");audit=audits[0];non=[t for t in audit["policy"]["trials"]if t["architecture"]is not None];best=max(non,key=lambda t:(t["eligible"],t["score"],t["auc_delta"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","hypothesis":HYPOTHESIS,"runtime":runtime,"market":"KR","fold":2,"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audit["inner_model_audit"],"outer_model_audit":audit["outer_model_audit"],"selection_locked_before_outer_evaluation":True,"nested_bootstrap_executed":False,"exact_entire_v69_fallback_contract":True,"output_written":False}),ensure_ascii=False,indent=2));return
        evaluation=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,smoke=args.smoke_test)
    non=[t for t in audits[0]["policy"]["trials"]if t["architecture"]is not None];best=max(non,key=lambda t:(t["eligible"],t["score"],t["auc_delta"]));summary={"status":"SMOKE_OK"if args.smoke_test else evaluation["status"],"hypothesis":HYPOTHESIS,"runtime":runtime,"folds_executed":len(audits),"smoke_market":args.smoke_market if args.smoke_test else None,"selected_models":[a["policy"]["selected"]["name"]for a in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"canonical_gate":evaluation["selected_summary"]["research_gate"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test:summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audits[0]["inner_model_audit"],"outer_model_audit":audits[0]["outer_model_audit"],"candidate_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]});print(json.dumps(clean(summary),ensure_ascii=False,indent=2));return
    result=write_outputs(args.output,authority,evidence,audits,evaluation);print(json.dumps(clean({**summary,"output_written":True,"controller":result}),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
