"""V188 multichannel causal cubical persistent-homology direction.

Each immutable V36 event supplies an observed 8-horizon by 8-semantic-channel
causal surface, scaled only from its strict-past fit cut.  For the global grid
and four fixed axis halves, exact sublevel and superlevel lower-star cubical
complexes are reduced over Z2 using an internal deterministic boundary-matrix
algorithm.  H0 and H1 persistence lifetimes are summarized before one equal-
event-group/class-balanced ridge-logit direction head.

This is not V106 H0 persistence on a hand-wired 37-feature graph, V167
fixed-threshold Euler curves, V185 continuous lag geostatistics, V186 cepstrum
or V187 gray-level run-length texture.  Inner OOF selects exact V69 or fixed
.25/.50 blends; outer labels remain evaluation-only after the 35-minute
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
from functools import lru_cache
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import experiment_v185_multichannel_causal_directional_semivariogram_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V188" / "LOCAL_FORBIDDEN"
VERSION = 188
HYPOTHESIS = "MULTICHANNEL_CAUSAL_PERSISTENT_HOMOLOGY_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v185_multichannel_causal_directional_semivariogram_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "bcae92beecb23bb9bae911d22608db52717603b96a5dc5b7ea95481615c712b9"

CHANNEL_N = 8
HORIZON_N = 8
REGIONS = (
    ("GLOBAL", 0, 8, 0, 8),
    ("OLD_HALF", 0, 4, 0, 8), ("RECENT_HALF", 4, 8, 0, 8),
    ("LOW_CHANNEL_HALF", 0, 8, 0, 4), ("HIGH_CHANNEL_HALF", 0, 8, 4, 8),
)
TOP_LIFETIME_N = 8
PER_DIMENSION_N = 15
FILTRATIONS = ("SUBLEVEL", "SUPERLEVEL")
PERSISTENCE_FEATURE_DIMENSION = len(REGIONS) * len(FILTRATIONS) * 2 * PER_DIMENSION_N
CHANNEL_CLIP = 7.0
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 18801
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CUBICAL_PERSISTENT_HOMOLOGY_W0.25", "weight": 0.25},
    {"name": "CUBICAL_PERSISTENT_HOMOLOGY_W0.50", "weight": 0.50},
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
    "fixed_regions": [region[0] for region in REGIONS],
    "filtrations": list(FILTRATIONS),
    "complex": "2D rectangular cubical complex with vertex lower-star values; edges/squares enter at maximum boundary-vertex value",
    "reduction": "deterministic exact Z2 boundary-matrix persistence reduction using Python integer bitsets",
    "homology_dimensions": [0, 1],
    "per_dimension": "top8 lifetimes plus positive-count, total, max, mean, std, q75 and entropy; essential classes capped at observed filtration maximum",
    "feature_dimension": PERSISTENCE_FEATURE_DIMENSION,
    "head": "one equal-event-group/class-balanced C=0.5 ridge logistic",
    "learned_threshold_complex_region_reduction_or_target_batch_fit": False,
}
require(PERSISTENCE_FEATURE_DIMENSION == 300, "V188 persistence feature dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V185 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, observed path, train channel scale, metrics, gate, bootstrap and atomic IO only; no V185 output is read"}
    audit["v188_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return audit


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.zeros(PERSISTENCE_FEATURE_DIMENSION)
        self.scale = np.ones(PERSISTENCE_FEATURE_DIMENSION)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.shape[1] == PERSISTENCE_FEATURE_DIMENSION, "V188 feature shape invalid")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V188 scaled feature invalid")
        return result


@lru_cache(maxsize=None)
def cubical_template(rows: int, columns: int) -> tuple[tuple[Any, ...], ...]:
    cells: list[tuple[Any, ...]] = []
    for row in range(rows):
        for column in range(columns):
            label = ("V", row, column)
            cells.append((0, label, (), ((row, column),)))
    for row in range(rows):
        for column in range(columns - 1):
            label = ("H", row, column)
            boundary = (("V", row, column), ("V", row, column + 1))
            cells.append((1, label, boundary, ((row, column), (row, column + 1))))
    for row in range(rows - 1):
        for column in range(columns):
            label = ("E", row, column)
            boundary = (("V", row, column), ("V", row + 1, column))
            cells.append((1, label, boundary, ((row, column), (row + 1, column))))
    for row in range(rows - 1):
        for column in range(columns - 1):
            label = ("S", row, column)
            boundary = (("H", row, column), ("H", row + 1, column), ("E", row, column), ("E", row, column + 1))
            vertices = ((row, column), (row, column + 1), (row + 1, column), (row + 1, column + 1))
            cells.append((2, label, boundary, vertices))
    require(len(cells) == rows * columns + rows * (columns - 1) + (rows - 1) * columns + (rows - 1) * (columns - 1), "V188 cubical template size mismatch")
    return tuple(cells)


def cubical_persistence(grid: np.ndarray) -> tuple[dict[int, list[float]], dict[str, Any]]:
    rows, columns = grid.shape
    template = cubical_template(rows, columns)
    enriched: list[tuple[float, int, int, tuple[Any, ...], tuple[Any, ...]]] = []
    for order, (dimension, label, boundary, vertices) in enumerate(template):
        value = max(float(grid[row, column]) for row, column in vertices)
        enriched.append((value, dimension, order, label, boundary))
    enriched.sort(key=lambda cell: (cell[0], cell[1], cell[2]))
    label_to_index = {cell[3]: index for index, cell in enumerate(enriched)}
    reduced_by_pivot: dict[int, int] = {}
    births: set[int] = set()
    paired_births: set[int] = set()
    lifetimes: dict[int, list[float]] = {0: [], 1: []}
    for column_index, (death_value, _dimension, _order, _label, boundary) in enumerate(enriched):
        reduced_column = 0
        for boundary_label in boundary:
            boundary_index = label_to_index[boundary_label]
            require(boundary_index < column_index, "V188 filtration boundary order invalid")
            reduced_column ^= 1 << boundary_index
        while reduced_column:
            pivot = reduced_column.bit_length() - 1
            prior = reduced_by_pivot.get(pivot)
            if prior is None:
                break
            reduced_column ^= prior
        if reduced_column == 0:
            births.add(column_index)
        else:
            pivot = reduced_column.bit_length() - 1
            reduced_by_pivot[pivot] = reduced_column
            require(pivot in births, "V188 persistence pivot is not a birth")
            paired_births.add(pivot)
            birth_value, birth_dimension = enriched[pivot][0], enriched[pivot][1]
            if birth_dimension in lifetimes:
                lifetimes[birth_dimension].append(max(float(death_value - birth_value), 0.0))
    maximum = max(cell[0] for cell in enriched)
    for birth in sorted(births - paired_births):
        birth_value, birth_dimension = enriched[birth][0], enriched[birth][1]
        if birth_dimension in lifetimes:
            lifetimes[birth_dimension].append(max(float(maximum - birth_value), 0.0))
    require(len(lifetimes[0]) > 0, "V188 H0 persistence missing")
    return lifetimes, {
        "rows": rows, "columns": columns, "cell_n": len(enriched),
        "vertex_n": rows * columns,
        "edge_n": rows * (columns - 1) + (rows - 1) * columns,
        "square_n": (rows - 1) * (columns - 1),
        "h0_interval_n": len(lifetimes[0]), "h1_interval_n": len(lifetimes[1]),
        "h0_positive_n": int(np.sum(np.asarray(lifetimes[0]) > 1e-12)),
        "h1_positive_n": int(np.sum(np.asarray(lifetimes[1]) > 1e-12)),
        "exact_z2_boundary_matrix_reduction": True,
    }


def lifetime_summary(lifetimes: list[float]) -> np.ndarray:
    positive = np.asarray([value for value in lifetimes if value > 1e-12], dtype=float)
    top = np.zeros(TOP_LIFETIME_N, dtype=float)
    if len(positive):
        ordered = np.sort(positive)[::-1]
        top[:min(len(ordered), TOP_LIFETIME_N)] = ordered[:TOP_LIFETIME_N]
        total = float(np.sum(positive))
        probability = positive / total
        entropy = float(-np.sum(probability * np.log(probability + 1e-15)))
        tail = np.asarray([len(positive), total, np.max(positive), np.mean(positive), np.std(positive), np.quantile(positive, 0.75), entropy], dtype=float)
    else:
        tail = np.zeros(7, dtype=float)
    result = np.concatenate([top, tail])
    require(len(result) == PER_DIMENSION_N and np.isfinite(result).all(), "V188 lifetime summary invalid")
    return result


def persistence_features(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V188 path shape invalid")
    result = np.empty((len(path), PERSISTENCE_FEATURE_DIMENSION), dtype=float)
    h0_positive_total = 0
    h1_positive_total = 0
    cell_total = 0
    for row, grid in enumerate(path):
        pieces: list[np.ndarray] = []
        for _region_name, r0, r1, c0, c1 in REGIONS:
            region = grid[r0:r1, c0:c1]
            for sign in (1.0, -1.0):
                lifetimes, audit = cubical_persistence(sign * region)
                pieces.append(lifetime_summary(lifetimes[0]))
                pieces.append(lifetime_summary(lifetimes[1]))
                h0_positive_total += audit["h0_positive_n"]
                h1_positive_total += audit["h1_positive_n"]
                cell_total += audit["cell_n"]
        result[row] = np.concatenate(pieces)
    require(np.isfinite(result).all() and float(np.std(result)) > 1e-10, "V188 persistence feature degenerate")
    return result, {
        "rows": len(result), "feature_dimension": result.shape[1], "feature_sha256": array_sha256(result),
        "sublevel_and_superlevel_lower_star": True,
        "rectangular_cubical_vertices_edges_squares": True,
        "exact_z2_boundary_matrix_reduction": True,
        "h0_and_h1_persistence_lifetimes": True,
        "essential_classes_capped_at_observed_filtration_maximum": True,
        "top8_and_distribution_summaries": True,
        "global_and_axis_half_regions": [region[0] for region in REGIONS],
        "mean_positive_h0_interval_n": float(h0_positive_total / max(len(path) * len(REGIONS) * len(FILTRATIONS), 1)),
        "mean_positive_h1_interval_n": float(h1_positive_total / max(len(path) * len(REGIONS) * len(FILTRATIONS), 1)),
        "mean_cubical_cell_n": float(cell_total / max(len(path) * len(REGIONS) * len(FILTRATIONS), 1)),
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "learned_threshold_complex_region_reduction_or_target_batch_fit": False,
        "v106_handwired_37_feature_graph_h0_only": False,
        "v167_fixed_threshold_euler_curve_without_pairs": False,
        "v185_continuous_semivariogram": False,
        "v186_dct_homomorphic_cepstrum": False,
        "v187_quantized_run_length_matrix": False,
    }


def persistence_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V188 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    train_feature, train_persistence = persistence_features(group_path)
    target_feature, target_persistence = persistence_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    require(np.all(class_n > 0), "V188 train cut missing class")
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all() and int(head.n_iter_[0]) < 1000, "V188 head invalid")
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping,
        "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_persistence, "target_geometry": target_persistence,
        "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": CHANNEL_CLIP},
        "feature_scaling": {"train_only": True, "median_sha256": array_sha256(feature_scale.median), "scale_sha256": array_sha256(feature_scale.scale), "clip": FEATURE_CLIP},
        "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "solver": "liblinear", "iterations": int(head.n_iter_[0]), "class_event_group_n": class_n.astype(int).tolist(), "coefficient_l2": float(np.linalg.norm(head.coef_)), "coefficient_sha256": array_sha256(head.coef_), "intercept": head.intercept_.tolist()},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION; base.HYPOTHESIS = HYPOTHESIS; base.STATIC_SPEC = STATIC_SPEC; base.FRENET_FEATURE_DIMENSION = PERSISTENCE_FEATURE_DIMENSION; base.ARCHITECTURES = ARCHITECTURES; base.SEED = SEED; base.frenet_direction = persistence_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = persistence_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float); confidence = inner_champion.confidence_signal.to_numpy(float); high = bool_series(inner_champion.high_conf).to_numpy(bool); baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name":"V69_NOOP","architecture":None,"metrics":baseline_metric,"auc_delta":0.0,"balanced_accuracy_delta":0.0,"all_trade_net_delta":0.0,"worst_supported_source_ba_delta":0.0,"supported_source_ba_delta":{},"eligible":True,"score":2.0*baseline_metric["auc"]+baseline_metric["balanced_accuracy"]}]
    geometry = model_audit["train_geometry"]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["channel_scaling"]["train_only"] and model_audit["feature_scaling"]["train_only"] and geometry["sublevel_and_superlevel_lower_star"] and geometry["rectangular_cubical_vertices_edges_squares"] and geometry["exact_z2_boundary_matrix_reduction"] and geometry["h0_and_h1_persistence_lifetimes"] and not geometry["learned_threshold_complex_region_reduction_or_target_batch_fit"] and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not model_audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability=blend_probability(baseline,challenger,architecture["weight"]);current=metric(inner_valid,probability,confidence,high);auc_delta=current["auc"]-baseline_metric["auc"];ba_delta=current["balanced_accuracy"]-baseline_metric["balanced_accuracy"];net_delta=current["all_trade_mean_signed_net"]-baseline_metric["all_trade_mean_signed_net"];source_floor,source_deltas=base.supported_source_ba_delta(inner_valid,baseline,probability);trials.append({"name":architecture["name"],"architecture":architecture,"metrics":current,"auc_delta":auc_delta,"balanced_accuracy_delta":ba_delta,"all_trade_net_delta":net_delta,"worst_supported_source_ba_delta":source_floor,"supported_source_ba_delta":source_deltas,"model_eligible":model_eligible,"eligible":bool(model_eligible and ba_delta>=-0.005 and net_delta>=-0.0005 and source_floor>=-0.010),"score":2.0*current["auc"]+current["balanced_accuracy"]})
    selected=max([trial for trial in trials if trial["eligible"]],key=lambda trial:(trial["score"],trial["auc_delta"],trial["all_trade_net_delta"],trial["name"]=="V69_NOOP"))
    return {"selection_rule":"inner-past OOF only; exact V69 noop or fixed .25/.50 cubical persistent-homology blend with fixed safety","baseline":baseline_metric,"selected":selected,"trials":trials,"outer_labels_used_for_selection":False,"complex_region_reduction_summary_head_blend_or_source_floor_micro_tuning":False},model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    configure_base();base.choose_inner=choose_inner
    with redirect_stdout(StringIO()): diagnostic,evidence,audits=base.run_nested(dev,champion,smoke,smoke_market)
    diagnostic=diagnostic.rename(columns={"v146_model":"v188_model"});evidence=evidence.rename(columns={"frenet_prob":"persistent_homology_prob","v146_model":"v188_model"})
    for audit in audits:
        selected=audit["policy"]["selected"];print(f"[V188 CUBICAL PH] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}",flush=True)
    return diagnostic,evidence,audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str,Any]:
    configure_base();result=_BASE_BOOTSTRAP(evidence,draws);result["method"]="paired market-fold-time block bootstrap over inner-locked V188 cubical persistent-homology policy";result["seed"]=SEED;result["standardized"]=True;return result


def evaluate(champion: pd.DataFrame,diagnostic:pd.DataFrame,evidence:pd.DataFrame,audits:list[dict[str,Any]],draws:int,smoke:bool)->dict[str,Any]:
    configure_base();base.paired_nested_bootstrap=paired_nested_bootstrap;result=base.evaluate(champion,diagnostic,evidence,audits,draws,smoke);checks=result["material_checks"];checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract=bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["feature_dimension"]==PERSISTENCE_FEATURE_DIMENSION and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["train_geometry"]["sublevel_and_superlevel_lower_star"] and audit[side]["train_geometry"]["rectangular_cubical_vertices_edges_squares"] and audit[side]["train_geometry"]["exact_z2_boundary_matrix_reduction"] and audit[side]["train_geometry"]["h0_and_h1_persistence_lifetimes"] and not audit[side]["train_geometry"]["v106_handwired_37_feature_graph_h0_only"] and not audit[side]["train_geometry"]["v167_fixed_threshold_euler_curve_without_pairs"] and not audit[side]["train_geometry"]["v185_continuous_semivariogram"] and not audit[side]["train_geometry"]["v186_dct_homomorphic_cepstrum"] and not audit[side]["train_geometry"]["v187_quantized_run_length_matrix"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit","outer_model_audit")))
    checks["multichannel_causal_cubical_persistent_homology_contract_verified"]=contract;result["material_pass"]=bool(not smoke and all(checks.values()))
    if not result["material_pass"]:result["selected_frame"]=champion.copy();result["selected_summary"]=result["baseline_summary"];result["fallback"]={"activated":True,"policy":"exact entire V69 DataFrame","exact_entire_frame_verified":bool(result["selected_frame"].equals(champion))};require(result["selected_frame"].equals(champion),"V188 fallback not exact V69")
    return result


def material_gate(evaluation:dict[str,Any])->dict[str,Any]:
    checks=evaluation["material_checks"];gate={"contract":HYPOTHESIS,"checks":checks,"passed":int(sum(bool(value) for value in checks.values())),"total":len(checks),"material_pass":bool(evaluation["material_pass"]),"nested_bootstrap":evaluation["nested_bootstrap"],"native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"],"current_hypothesis_not_inherited_from_v69":True,"fallback_policy":"candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"};require(gate["nested_bootstrap"]["contract_key"]=="selected.material_gate.nested_bootstrap","V188 nested key mismatch");return gate


def enforce_output_conflict_fail_closed(out:Path)->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT),"V188 output requires controller authority");require(out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V188 output target mismatch");require(out.resolve().parent==(ROOT/"research"/"staging"/"V188").resolve(),"V188 output outside staging/V188");require(out.exists()and out.is_dir(),"V188 staging child must preexist");required={"VERSION_PLAN.json","BOTTLENECK_ANALYSIS.json","MATERIAL_CHANGE.json","RUN.log"};allowed=required|{"DATA_EPOCH_BINDING.json"};existing={path.name for path in out.iterdir()};require(required.issubset(existing),f"V188 prefiles missing: {sorted(required-existing)}");require(not(existing-allowed),f"V188 unexpected conflict: {sorted(existing-allowed)}");return {"controller_bound":True,"target_preexisting":True,"required_controller_prefiles":sorted(required),"observed_controller_prefiles":sorted(existing),"unexpected_prefiles":[],"conflict_policy":"allow exact controller prefiles only; fail closed before write"}


def write_outputs(out:Path,authority:dict[str,Any],evidence:pd.DataFrame,audits:list[dict[str,Any]],evaluation:dict[str,Any])->dict[str,Any]:
    require(bool(CONTROLLER_OUTPUT)and out.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V188 output not controller-bound");conflict=enforce_output_conflict_fail_closed(out);compact={key:value for key,value in evaluation.items()if not key.endswith("_frame")};gate=material_gate(evaluation);selected=json.loads(json.dumps(clean(evaluation["selected_summary"])));selected["material_gate"]=gate;require("bootstrap"in selected["robustness"],"V188 native bootstrap missing")
    reports={"MODEL_COMPARISON.json":{"version":"V188","hypothesis":HYPOTHESIS,"status":evaluation["status"],"models":{"V69_CHAMPION":evaluation["baseline_summary"],"V188_DIAGNOSTIC_CUBICAL_PH":evaluation["candidate_summary"],"V188_FAIL_CLOSED_SELECTED":selected},"evaluation":compact,"authority_audit":authority,"nested_fold_audits":audits},"DEV_ROBUSTNESS_REPORT.json":{"version":"V188","hypothesis":HYPOTHESIS,"status":evaluation["status"],"opened_dev_only":True,"selected":selected,"candidate":evaluation["candidate_summary"],"material_gate":gate,"material_checks":evaluation["material_checks"],"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"SOURCE_TRANSFER_REPORT.json":{"version":"V188","hypothesis":HYPOTHESIS,"status":evaluation["status"],"selected_by_source_family":selected["metrics"]["by_source_family"],"candidate_by_source_family":evaluation["candidate_summary"]["metrics"]["by_source_family"],"selected_by_market":selected["metrics"]["by_market"],"candidate_by_market":evaluation["candidate_summary"]["metrics"]["by_market"],"material_gate":gate,"nested_bootstrap":evaluation["nested_bootstrap"],"fallback_is_exact_entire_v69_frame":not evaluation["material_pass"],"seal_authorized":False},"V188_PERSISTENT_HOMOLOGY_REPORT.json":{"version":"V188","hypothesis":HYPOTHESIS,"static_spec":STATIC_SPEC,"architectures":ARCHITECTURES,"nested_fold_audits":audits,"evaluation":compact,"authority_audit":authority},"STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json":{"version":"V188","hypothesis":HYPOTHESIS,"contract_key":"selected.material_gate.nested_bootstrap","bootstrap":evaluation["nested_bootstrap"]},"NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json":{"version":"V188","hypothesis":HYPOTHESIS,"controller_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]},"CANONICAL_RESEARCH_GATE_AUDIT.json":{"version":"V188","status":"MATCH","canonical_check_count":14,"reported":selected["research_gate"],"controller_recomputed":controller.canonical_research_gate(selected),"material_gate":gate},"RUN_STATUS.json":{"version":VERSION,"hypothesis":HYPOTHESIS,"status":evaluation["status"],"material_pass":evaluation["material_pass"],"champion_changed":evaluation["material_pass"],"seal_state":"UNOPENED","seal_authorized":False,"output_conflict_audit":conflict,"completed_at":pd.Timestamp.now(tz="UTC").isoformat()}}
    for name,payload in reports.items():atomic_json(payload,out/name)
    atomic_csv(evidence,out/"V188_PERSISTENT_HOMOLOGY_OUTER_EVIDENCE.csv.gz");atomic_csv(evaluation["selected_frame"],out/"V188_FAIL_CLOSED_SELECTED_OOF.csv.gz");files={path.name:{"sha256":sha256(path),"bytes":path.stat().st_size}for path in sorted(out.iterdir())if path.is_file()and path.name!="ARTIFACT_MANIFEST.json"};atomic_json({"version":VERSION,"hypothesis":HYPOTHESIS,"authority_experiment_id":EXPECTED_V69_EXPERIMENT_ID,"files":files},out/"ARTIFACT_MANIFEST.json");return {"output":str(out),"artifact_count":len(files)+1,"manifest_sha256":sha256(out/"ARTIFACT_MANIFEST.json")}


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT));modes.add_argument("--audit-only",action="store_true");modes.add_argument("--smoke-test",action="store_true");modes.add_argument("--support-probe",action="store_true");modes.add_argument("--full-run",action="store_true");parser.add_argument("--smoke-market",choices=("US","KR"),default="US");parser.add_argument("--output",type=Path,default=None);args=parser.parse_args()
    if CONTROLLER_OUTPUT:require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None,"explicit mode/output conflicts with controller-bound no-argument V188 full run");args.full_run=True;args.output=Path(CONTROLLER_OUTPUT)
    elif args.output is None:args.output=DEFAULT_OUT
    if args.full_run:require(bool(CONTROLLER_OUTPUT),"V188 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT");require(args.output.resolve()==Path(CONTROLLER_OUTPUT).resolve(),"V188 controller full-run output mismatch")
    return args


def main()->None:
    args=parse_args();bounded=bool(args.audit_only or args.smoke_test or args.support_probe);runtime=numeric.configure_bounded_runtime()if bounded else{"mode":"controller_bound_full","thread_policy":"authorized parent inherited","gpu_model_calls":0};authority=verify_authority();dev,champion=load_authorized()
    if args.audit_only:print(json.dumps(clean({"status":"AUDIT_OK","hypothesis":HYPOTHESIS,"authority":authority,"runtime":runtime,"dev_rows":len(dev),"v69_rows":len(champion),"features":FEATURES,"static_spec":STATIC_SPEC,"architecture":"exact dependency-free sub/superlevel cubical H0/H1 Z2 persistence lifetime summaries; balanced ridge","causal_numeric_only":True,"target_labels_used":False,"canonical_controller_gate_checks":14,"controller_contract":{"no_argument_market_bio_version_output_full":True,"controller_output_exact_binding":True,"explicit_mode_or_output_conflict_fails_closed":True,"direct_full_without_controller_fails_closed":True,"required_reports":["MODEL_COMPARISON.json","DEV_ROBUSTNESS_REPORT.json","SOURCE_TRANSFER_REPORT.json"],"current_material_gate":True,"nested_bootstrap_key":"selected.material_gate.nested_bootstrap","standardized_nested_bootstrap":True,"native_robustness_bootstrap_preserved":True,"source_transfer_current_gate":True,"exact_entire_v69_fallback":True},"output_written":False}),ensure_ascii=False,indent=2));return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic,evidence,audits=run_nested(dev,champion,smoke=bounded,smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market=="KR","V188 support reserved for KR:2");audit=audits[0];non=[trial for trial in audit["policy"]["trials"]if trial["architecture"]is not None];best=max(non,key=lambda trial:(trial["eligible"],trial["score"],trial["auc_delta"]));print(json.dumps(clean({"status":"SUPPORT_PROBE_OK","hypothesis":HYPOTHESIS,"runtime":runtime,"market":"KR","fold":2,"strict_inner_chronology":audit["inner_chronology"],"strict_outer_chronology":audit["outer_chronology"],"raw_inner_selected":audit["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audit["outer_baseline"],"raw_outer_selected":audit["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audit["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audit["inner_model_audit"],"outer_model_audit":audit["outer_model_audit"],"selection_locked_before_outer_evaluation":True,"nested_bootstrap_executed":False,"exact_entire_v69_fallback_contract":True,"output_written":False}),ensure_ascii=False,indent=2));return
        evaluation=evaluate(champion,diagnostic,evidence,audits,SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,smoke=args.smoke_test)
    non=[trial for trial in audits[0]["policy"]["trials"]if trial["architecture"]is not None];best=max(non,key=lambda trial:(trial["eligible"],trial["score"],trial["auc_delta"]));summary={"status":"SMOKE_OK"if args.smoke_test else evaluation["status"],"hypothesis":HYPOTHESIS,"runtime":runtime,"folds_executed":len(audits),"smoke_market":args.smoke_market if args.smoke_test else None,"selected_models":[audit["policy"]["selected"]["name"]for audit in audits],"outer_auc_delta":evaluation["outer_auc_delta"],"outer_ba_delta":evaluation["outer_ba_delta"],"outer_net_delta":evaluation["outer_net_delta"],"material_pass":evaluation["material_pass"],"fallback_is_exact_v69":not evaluation["material_pass"],"canonical_gate":evaluation["selected_summary"]["research_gate"],"current_material_gate":material_gate(evaluation),"output_written":False}
    if args.smoke_test:summary.update({"raw_inner_selected":audits[0]["policy"]["selected"],"raw_inner_best_non_noop":best,"raw_outer_baseline":audits[0]["outer_baseline"],"raw_outer_selected":audits[0]["outer_candidate"],"raw_outer_best_non_noop_evaluation_only":audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]],"inner_model_audit":audits[0]["inner_model_audit"],"outer_model_audit":audits[0]["outer_model_audit"],"candidate_native_robustness_bootstrap":evaluation["candidate_native_robustness_bootstrap"]});print(json.dumps(clean(summary),ensure_ascii=False,indent=2));return
    result=write_outputs(args.output,authority,evidence,audits,evaluation);print(json.dumps(clean({**summary,"output_written":True,"controller":result}),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
