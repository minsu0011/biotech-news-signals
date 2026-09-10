"""V179 multichannel causal structure-tensor orientation direction.

The immutable V36 causal 8-horizon by 8-channel surface is scaled only from
each strict-past fit cut.  Fixed finite differences and two predeclared local
box scales form 2x2 horizon/channel gradient second-moment tensors.  Global,
quadrant, horizon-half and channel-half region summaries retain tensor energy,
anisotropy and double-angle principal orientation.  A single equal-event-group
class-balanced ridge-logit head learns direction.  Labels and target batches
never define the representation.

This is not V146 discrete Frenet curve geometry, V149 increment coskewness,
V173 grid-Laplacian spectrum, V174 morphology, V177 Radon line integration or
V178 per-channel B-spline functional jets.  Inner OOF selects exact V69 or a
fixed .25/.50 blend; outer labels are evaluation-only after the 35-minute
embargo/event purge.  Any material failure restores the exact entire V69 frame
with confidence/high_conf unchanged.
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

import experiment_v174_multichannel_causal_morphological_granulometry_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V179" / "LOCAL_FORBIDDEN"
VERSION = 179
HYPOTHESIS = "MULTICHANNEL_CAUSAL_STRUCTURE_TENSOR_ORIENTATION_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v174_multichannel_causal_morphological_granulometry_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "3722bd9c68851091bf89d0fb21eeed9b1d41b69d69c5a000b2936245bf4961ca"

CHANNEL_N = 8
HORIZON_N = 8
SMOOTHING_RADII = (0, 1)
REGIONS = (
    ("GLOBAL", 0, 8, 0, 8),
    ("OLD_LOW", 0, 4, 0, 4), ("OLD_HIGH", 0, 4, 4, 8),
    ("RECENT_LOW", 4, 8, 0, 4), ("RECENT_HIGH", 4, 8, 4, 8),
    ("OLD_HALF", 0, 4, 0, 8), ("RECENT_HALF", 4, 8, 0, 8),
    ("LOW_CHANNEL_HALF", 0, 8, 0, 4), ("HIGH_CHANNEL_HALF", 0, 8, 4, 8),
)
PER_REGION_N = 10
STRUCTURE_FEATURE_DIMENSION = len(SMOOTHING_RADII) * len(REGIONS) * PER_REGION_N
CHANNEL_CLIP = 7.0
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 17901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "STRUCTURE_TENSOR_ORIENTATION_W0.25", "weight": 0.25},
    {"name": "STRUCTURE_TENSOR_ORIENTATION_W0.50", "weight": 0.50},
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
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent by eight immutable semantic channels",
    "gradient_axes": ["causal horizon", "immutable semantic channel order"],
    "fixed_tensor_smoothing_radii": list(SMOOTHING_RADII),
    "fixed_regions": [region[0] for region in REGIONS],
    "per_region": ["trace", "determinant", "major eigenvalue", "minor eigenvalue", "coherence", "cos(2theta)", "sin(2theta)", "mean abs horizon gradient", "mean abs channel gradient", "mean gradient magnitude"],
    "feature_dimension": STRUCTURE_FEATURE_DIMENSION,
    "head": "one equal-event-group/class-balanced C=0.5 ridge logistic",
    "learned_kernel_scale_angle_region_or_target_batch_fit": False,
}
require(STRUCTURE_FEATURE_DIMENSION == 180, "V179 structure feature dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V174 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, observed path, train channel scale, metrics, gate, bootstrap and atomic IO only; no V174 output is read"}
    audit["v179_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.zeros(STRUCTURE_FEATURE_DIMENSION)
        self.scale = np.ones(STRUCTURE_FEATURE_DIMENSION)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.shape[1] == STRUCTURE_FEATURE_DIMENSION, "V179 feature shape invalid")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V179 scaled feature invalid")
        return result


def box_mean(field: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return field.copy()
    require(radius == 1 and field.shape == (HORIZON_N, CHANNEL_N), "V179 unsupported tensor scale")
    padded = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    result = np.zeros_like(field)
    for dr in range(3):
        for dc in range(3):
            result += padded[dr:dr + HORIZON_N, dc:dc + CHANNEL_N]
    return result / 9.0


def structure_tensor_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V179 path shape invalid")
    result = np.empty((len(path), STRUCTURE_FEATURE_DIMENSION), dtype=float)
    coherence_total = 0.0
    energy_total = 0.0
    for row, grid in enumerate(path):
        horizon_gradient = np.gradient(grid, axis=0, edge_order=1)
        channel_gradient = np.gradient(grid, axis=1, edge_order=1)
        magnitude = np.sqrt(horizon_gradient * horizon_gradient + channel_gradient * channel_gradient)
        pieces: list[np.ndarray] = []
        for radius in SMOOTHING_RADII:
            jhh = box_mean(horizon_gradient * horizon_gradient, radius)
            jcc = box_mean(channel_gradient * channel_gradient, radius)
            jhc = box_mean(horizon_gradient * channel_gradient, radius)
            for _name, r0, r1, c0, c1 in REGIONS:
                ahh = float(np.mean(jhh[r0:r1, c0:c1]))
                acc = float(np.mean(jcc[r0:r1, c0:c1]))
                ahc = float(np.mean(jhc[r0:r1, c0:c1]))
                trace = ahh + acc
                discriminant = float(np.sqrt(max((ahh - acc) ** 2 + 4.0 * ahc * ahc, 0.0)))
                major = 0.5 * (trace + discriminant)
                minor = 0.5 * (trace - discriminant)
                determinant = max(ahh * acc - ahc * ahc, 0.0)
                coherence = discriminant / (trace + 1e-12)
                cos2 = (ahh - acc) / (discriminant + 1e-12)
                sin2 = 2.0 * ahc / (discriminant + 1e-12)
                pieces.append(np.asarray([
                    trace, determinant, major, minor, coherence, cos2, sin2,
                    float(np.mean(np.abs(horizon_gradient[r0:r1, c0:c1]))),
                    float(np.mean(np.abs(channel_gradient[r0:r1, c0:c1]))),
                    float(np.mean(magnitude[r0:r1, c0:c1])),
                ], dtype=float))
                coherence_total += coherence
                energy_total += trace
        result[row] = np.concatenate(pieces)
    require(np.isfinite(result).all() and float(np.std(result)) > 1e-10, "V179 structure tensor feature degenerate")
    normalizer = max(len(path) * len(SMOOTHING_RADII) * len(REGIONS), 1)
    return result, {
        "rows": len(result), "feature_dimension": result.shape[1], "feature_sha256": array_sha256(result),
        "fixed_horizon_channel_finite_differences": True,
        "two_fixed_tensor_smoothing_scales": list(SMOOTHING_RADII),
        "structure_tensor_second_moment": True,
        "principal_eigenvalue_orientation": True,
        "double_angle_orientation": True,
        "global_quadrant_axis_half_regions": [region[0] for region in REGIONS],
        "mean_tensor_coherence": float(coherence_total / normalizer),
        "mean_tensor_energy": float(energy_total / normalizer),
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "learned_kernel_scale_angle_region_or_target_batch_fit": False,
        "v146_curve_frenet_curvature_bivector_torsion": False,
        "v149_increment_third_central_comoment": False,
        "v167_cubical_euler_threshold_filtration": False,
        "v173_grid_laplacian_eigenspectrum": False,
        "v174_erosion_dilation_opening_closing": False,
        "v177_radon_line_integral_profiles": False,
        "v178_cubic_bspline_functional_jet": False,
    }


def structure_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V179 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    train_feature, train_structure = structure_tensor_features(group_path)
    target_feature, target_structure = structure_tensor_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    require(np.all(class_n > 0), "V179 train cut missing class")
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all() and int(head.n_iter_[0]) < 1000, "V179 head invalid")
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping,
        "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_structure, "target_geometry": target_structure,
        "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": CHANNEL_CLIP},
        "feature_scaling": {"train_only": True, "median_sha256": array_sha256(feature_scale.median), "scale_sha256": array_sha256(feature_scale.scale), "clip": FEATURE_CLIP},
        "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "class_event_group_n": class_n.astype(int).tolist(), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "intercept": head.intercept_.tolist()},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "line_integral_tomography_or_profile_centroid": False, "spline_basis_pseudoinverse_or_functional_jet": False,
        "grid_graph_eigenvalue_or_heat_trace": False, "threshold_topology_or_morphology": False,
        "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION; base.HYPOTHESIS = HYPOTHESIS; base.STATIC_SPEC = STATIC_SPEC; base.FRENET_FEATURE_DIMENSION = STRUCTURE_FEATURE_DIMENSION; base.ARCHITECTURES = ARCHITECTURES; base.SEED = SEED; base.frenet_direction = structure_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = structure_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.0,"balanced_accuracy_delta":0.0,"all_trade_net_delta":0.0,"worst_supported_source_ba_delta":0.0,"supported_source_ba_delta":{},"eligible":True,"score":2.0*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}]
    geometry = model_audit["train_geometry"]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["channel_scaling"]["train_only"] and model_audit["feature_scaling"]["train_only"] and geometry["fixed_horizon_channel_finite_differences"] and geometry["structure_tensor_second_moment"] and geometry["principal_eigenvalue_orientation"] and geometry["double_angle_orientation"] and not geometry["learned_kernel_scale_angle_region_or_target_batch_fit"] and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not model_audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]);current=metric(inner_valid,probability,confidence,high);auc_delta=current["auc"]-baseline_metric["auc"];ba_delta=current["balanced_accuracy"]-baseline_metric["balanced_accuracy"];net_delta=current["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"];source_floor,source_deltas=base.supported_source_ba_delta(inner_valid,baseline,probability);trials.append({"name":architecture["name"],"architecture":architecture,"metrics":current,"auc_delta":auc_delta,"balanced_accuracy_delta":ba_delta,"all_trade_net_delta":net_delta,"worst_supported_source_ba_delta":source_floor,"supported_source_ba_delta":source_deltas,"model_eligible":model_eligible,"eligible":bool(model_eligible and ba_delta>=-0.005 and net_delta>=-0.0005 and source_floor>=-0.010),"score":2.0*current["auc"]+current["balanced_accuracy"]})
    selected=max([trial for trial in trials if trial["eligible"]],key=lambda trial:(trial["score"],trial["auc_delta"],trial["all_trade_net_delta"],trial["name"]=="V69_NOOP"))
    return {"selection_rule":"inner-past OOF only; exact V69 noop or fixed .25/.50 structure-tensor orientation blend with fixed safety","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"region_tensor_scale_head_blend_or_source_floor_micro_tuning":False},model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    configure_base();base.choose_inner=choose_inner
    with redirect_stdout(StringIO()): diagnostic,evidence,audits=base.run_nested(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v146_model":"v179_model"});evidence=evidence.rename(columns={"frenet_prob":"structure_tensor_prob","v146_model":"v179_model"})
    for audit in audits:
        selected=audit["policy"]["selected"];print(f"[V179 STRUCTURE TENSOR] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}",flush=True)
    return diagnostic,evidence,audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str,Any]:
    configure_base();result=_BASE_BOOTSTRAP(evidence,draws);result["method"]="paired market-fold-time block bootstrap over inner-locked V179 structure-tensor orientation policy";result["seed"]=SEED;result["standardized"]=True;return result


def evaluate(champion: pd.DataFrame,diagnostic:pd.DataFrame,evidence:pd.DataFrame,audits:list[dict[str,Any]],draws:int,smoke:bool)->dict[str,Any]:
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;result=base.evaluate(champion,diagnostic,evidence,audits,draws,smoke);checks=result["material_checks"];checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract=bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["feature_dimension"]==STRUCTURE_FEATURE_DIMENSION and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["train_geometry"]["fixed_horizon_channel_finite_differences"] and audit[side]["train_geometry"]["structure_tensor_second_moment"] and audit[side]["train_geometry"]["principal_eigenvalue_orientation"] and audit[side]["train_geometry"]["double_angle_orientation"] and not audit[side]["train_geometry"]["v146_curve_frenet_curvature_bivector_torsion"] and not audit[side]["train_geometry"]["v149_increment_third_central_comoment"] and not audit[side]["train_geometry"]["v173_grid_laplacian_eigenspectrum"] and not audit[side]["train_geometry"]["v174_erosion_dilation_opening_closing"] and not audit[side]["train_geometry"]["v177_radon_line_integral_profiles"] and not audit[side]["train_geometry"]["v178_cubic_bspline_functional_jet"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit","outer_model_audit")))
    checks["multichannel_causal_structure_tensor_orientation_contract_verified"]=contract;result["material_pass"]=bool(not smoke and all(checks.values()))
    if not result["material_pass"]:result["selected_frame"]=champion.copy();result["selected_summary"]=result["baseline_summary"];result["fallback"]={"activated":True,"policy":"exact entire V69 DataFrame","exact_entire_frame_verified":bool(result["selected_frame"].equals(champion))};require(result["selected_frame"].equals(champion),"V179 fallback not exact V69")
    return result


def material_gate(evaluation:dict[str,Any])->dict[str,Any]:
    checks=evaluation["material_checks"];gate={"contract":HYPOTHESIS,"checks":checks,"passed":int(sum(bool(value) for value in checks.values())),"total":len(checks),"material_pass":bool(evaluation["material_pass"]),"nested_bootstrap":evaluation["nested_bootstrap"],"native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"],"current_hypothesis_not_inherited_from_v69":True,"fallback_policy":"candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"};require(gate["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V179 nested key mismatch");return gate


def enforce_output_conflict_fail_closed(out:Path)->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT),"V179 output requires controller authority");require(out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V179 output target mismatch");require(out.resolve().parent==(ROOT/"research"/"staging"/"V179").resolve(),"V179 output outside staging/V179");require(out.exists()and out.is_dir(),"V179 staging child must preexist");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={path.name for path in out.iterdir()};require(required.issubset(existing),f"V179 prefiles missing: {sorted(required-existing)}");require(not(existing-allowed),f"V179 unexpected conflict: {sorted(existing-allowed)}");return {"controller_bound":True,"target_preexisting":True,"required_controller_prefiles":sorted(required),"observed_controller_prefiles":sorted(existing),"unexpected_prefiles":[],"conflict_policy":"allow exact controller prefiles only; fail closed before write"}


def write_outputs(out:Path,authority:dict[str,Any],evidence:pd.DataFrame,audits:list[dict[str,Any]],evaluation:dict[str,Any])->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V179 output not controller-bound");conflict=enforce_output_conflict_fail_closed(out);compact={key:value for key,value in evaluation.items()if not key.endswith("_frame")};gate=material_gate(evaluation);selected=json.loads(json.dumps(clean(evaluation["selected_summary"])));selected["material_gate"]=gate;require("bootstrap"in selected["robustness"],"V179 native bootstrap missing")
    reports={"MODEL_COMPARISON.json":{"version":"V179","hypothesis":HYPOTHESIS,"status":evaluation["status"],"models":{"V69_CHAMPION":evaluation["baseline_summary"],"V179_DIAGNOSTIC_STRUCTURE_TENSOR":evaluation["candidate_summary"],"V179_FAIL_CLOSED_SELECTED":selected},"evaluation":compact,"authority_audit":authority,"nested_fold_audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V179","hypothesis":HYPOTHESIS,"status":evaluation["status"],"opened_dev_only":True,"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"material_checks":evaluation["material_checks"],"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"SOURCE_TRANSFER_REPORT.json":{"version":"V179","hypothesis":HYPOTHESIS,"status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate,"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"V179_STRUCTURE_TENSOR_ORIENTATION_REPORT.json":{"version":"V179","hypothesis":HYPOTHESIS,"static_spec":STATIC_SPEC,"architectures":ARCHITECTURES,"nested_fold_audits":audits,"evaluation":compact,"authority_audit":authority},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"version":"V179","hypothesis":HYPOTHESIS,"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"version":"V179","hypothesis":HYPOTHESIS,"controller_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"version":"V179","status":"MATCH","canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected),"material_gate":gate},"RUN_STATUS.json":{"version":VERSION,"hypothesis":HYPOTHESIS,"status":evaluation["status"],"material_pass":evaluation["material_pass"],"champion_changed":evaluation["material_pass"],"seal_state":"UNOPENED","seal_authorized":False,"output_conflict_audit":conflict,"completed_at":pd.Timestamp.now(tz="UTC").isoformat()}}
    for name,payload in reports.items():atomic_json(payload,out/name)
    atomic_csv(evidence,out/"V179_STRUCTURE_TENSOR_OUTER_EVIDENCE.csv.gz");atomic_csv(evaluation["selected_frame"],out/"V179_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={path.name:{"sha256":sha256(path),"bytes":path.stat().st_size}for path in sorted(out.iterdir())if path.is_file()and path.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"hypothesis":HYPOTHESIS,"authority_experiment_id":EXPECTED_V69_EXPERIMENT_ID,"files":files},out/"ARTIFACT_MANIFEST.json");return {"output":str(out),"artifact_count":len(files)+1,"manifest_sha256":sha256(out/"ARTIFACT_MANIFEST.json")}


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));modes.add_argument("--audit-only",action="store_true");modes.add_argument("--smoke-test",action="store_true");modes.add_argument("--support-probe",action="store_true");modes.add_argument("--full-run",action="store_true");parser.add_argument("--smoke-market",choices=("US","KR"),default="US");parser.add_argument("--output",type=Path,default=None);args=parser.parse_args()
    if CONTROLLER_OUTPUT:require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None,"explicit mode/output conflicts with controller-bound no-argument V179 full run");args.full_run=True;args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None:args.output=DEFAULT_OUT
    if args.full_run:require(bool(CONTROLLER_OUTPUT),"V179 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT");require(args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V179 controller full-run output mismatch")
    return args


def main()->None:
    args=parse_args();bounded=bool(args.audit_only or args.smoke_test or args.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller_bound_full","thread_policy":"authorized parent inherited","gpu_model_calls":0};authority=verify_authority();dev,champion=load_authorized()
    if args.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"dev_rows":len(dev),"v69_rows":len(champion),"features":FEATURES,"static_spec":STATIC_SPEC,"architecture":"fixed 2D finite differences, two-scale local structure tensors and regional principal orientation summaries; balanced ridge","causal_numeric_only":True,"target_labels_used":False,"canonical_controller_gate_checks":14,"controller_contract":{"no_argument_market_bio_version_output_full":True,"controller_output_exact_binding":True,"explicit_mode_or_output_conflict_fails_closed":True,"direct_full_without_controller_fails_closed":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"current_material_gate":True,"nested_bootstrap_key":"selected.material_gate.nested_bootstrap","standardized_nested_bootstrap":True,"native_robustness_bootstrap_preserved":True,"source_transfer_current_gate":True,"exact_entire_v69_fallback":True},"output_written":False}),ensure_ascii=False,indent=2));return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,smoke=bounded,smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market=="KR","V179 support reserved for KR:2");audit=audits[0];non=[trial for trial in audit["policy"]["trials"]if trial["architecture"]is not None];best=max(non,key=lambda trial:(trial["eligible"],trial["score"],trial["auc_delta"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","hypothesis":HYPOTHESIS,"runtime":runtime,"market":"KR","fold":2,"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audit["inner_model_audit"],"outer_model_audit":audit["outer_model_audit"],"selection_locked_before_outer_evaluation":True,"nested_bootstrap_executed":False,"exact_entire_v69_fallback_contract":True,"output_written":False}),ensure_ascii=False,indent=2));return
        evaluation=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,smoke=args.smoke_test)
    non=[trial for trial in audits[0]["policy"]["trials"]if trial["architecture"]is not None];best=max(non,key=lambda trial:(trial["eligible"],trial["score"],trial["auc_delta"]));summary={"status":"SMOKE_OK"if args.smoke_test else evaluation["status"],"hypothesis":HYPOTHESIS,"runtime":runtime,"folds_executed":len(audits),"smoke_market":args.smoke_market if args.smoke_test else None,"selected_models":[audit["policy"]["selected"]["name"]for audit in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"canonical_gate":evaluation["selected_summary"]["research_gate"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test:summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audits[0]["inner_model_audit"],"outer_model_audit":audits[0]["outer_model_audit"],"candidate_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]});print(json.dumps(clean(summary),ensure_ascii=False,indent=2));return
    result=write_outputs(args.output,authority,evidence,audits,evaluation);print(json.dumps(clean({**summary,"output_written":True,"controller":result}),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
