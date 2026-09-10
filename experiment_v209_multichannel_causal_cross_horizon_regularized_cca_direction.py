"""V209 strict-train cross-horizon regularized CCA direction."""
from __future__ import annotations
import os
import ctypes

CPU_AFFINITY_MASK = 0xC0000000
if os.name == "nt":
    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    _kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
    _kernel32.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    _kernel32.GetProcessAffinityMask.restype = ctypes.c_bool
    _process = _kernel32.GetCurrentProcess()
    if not _kernel32.SetProcessAffinityMask(_process, ctypes.c_size_t(CPU_AFFINITY_MASK)):
        raise RuntimeError("V209 pre-import CPU30-31 pin failed")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import experiment_v204_multichannel_causal_two_additive_choquet_capacity_direction as scaffold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.name == "nt" and not _kernel32.SetProcessAffinityMask(_process, ctypes.c_size_t(CPU_AFFINITY_MASK)):
    raise RuntimeError("V209 post-scaffold CPU30-31 restore failed")

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
base = scaffold.base
ROOT = scaffold.ROOT
VERSION = 209
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CROSS_HORIZON_REGULARIZED_CCA_DIRECTION_V1"
DEFAULT_OUT = ROOT / "research" / "staging" / "V209" / "LOCAL_FORBIDDEN"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v204_multichannel_causal_two_additive_choquet_capacity_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "a1f6b47244c46bb3f5679e8198cd95499f91dc952cf30066e25d98e7e847588e"

CCA_RANK = 8
CCA_RIDGE = 0.10
FEATURE_DIMENSION = 40
RIDGE_LOGISTIC_C = 0.5
FEATURE_CLIP = 8.0
SEED = 20901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
ARCHITECTURES = (
    {"name": "CROSS_HORIZON_CCA_W0.25", "weight": 0.25},
    {"name": "CROSS_HORIZON_CCA_W0.50", "weight": 0.50},
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

STATIC_SPEC = {
    "path_shape": [8, 8],
    "views": "fixed past half horizons 120/60/30/15 versus recent half 10/5/2/1, 32 coordinates each",
    "fit": "strict-train label-free ridge-whitened cross-covariance SVD",
    "cca_rank": CCA_RANK,
    "cca_ridge": CCA_RIDGE,
    "features": "past and recent canonical coordinates, pair products, signed differences and absolute differences",
    "feature_dimension": FEATURE_DIMENSION,
    "head": "equal-event-group/class-balanced C=.5 ridge logistic",
    "target_model_fit": False,
    "gpu_model_calls": 0,
}


def observed_affinity():
    if os.name != "nt":
        return {"mask_hex": None, "cpus": None}
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    require(_kernel32.GetProcessAffinityMask(_process, ctypes.byref(process_mask), ctypes.byref(system_mask)), "V209 affinity query failed")
    cpus = [index for index in range(64) if process_mask.value & (1 << index)]
    require(process_mask.value == CPU_AFFINITY_MASK and cpus == [30, 31], "V209 affinity drift")
    return {"mask_hex": f"0x{process_mask.value:08X}", "cpus": cpus}


def verify_authority():
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "V204 scaffold changed")
    authority = scaffold.verify_authority()
    authority["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "authority/chronology/path/metrics/gate/bootstrap/IO only; no V204 output"}
    authority["v209_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return authority


def inverse_sqrt(matrix):
    value, vector = np.linalg.eigh((matrix + matrix.T) / 2.0)
    value = np.maximum(value, CCA_RIDGE)
    return (vector * (1.0 / np.sqrt(value))) @ vector.T, value


def fit_cca(path):
    require(path.ndim == 3 and path.shape[1:] == (8, 8), "V209 path shape")
    past = path[:, :, :4].reshape(len(path), 32)
    recent = path[:, :, 4:].reshape(len(path), 32)
    past_mean = past.mean(axis=0)
    recent_mean = recent.mean(axis=0)
    x = past - past_mean
    y = recent - recent_mean
    denominator = max(len(path) - 1, 1)
    cxx = x.T @ x / denominator + CCA_RIDGE * np.eye(32)
    cyy = y.T @ y / denominator + CCA_RIDGE * np.eye(32)
    cxy = x.T @ y / denominator
    wx, x_eigen = inverse_sqrt(cxx)
    wy, y_eigen = inverse_sqrt(cyy)
    left, singular, right_t = np.linalg.svd(wx @ cxy @ wy, full_matrices=False)
    past_loading = wx @ left[:, :CCA_RANK]
    recent_loading = wy @ right_t.T[:, :CCA_RANK]
    for mode in range(CCA_RANK):
        anchor = int(np.argmax(np.abs(past_loading[:, mode])))
        if past_loading[anchor, mode] < 0:
            past_loading[:, mode] *= -1.0
            recent_loading[:, mode] *= -1.0
    require(np.isfinite(past_loading).all() and np.isfinite(recent_loading).all(), "V209 CCA nonfinite")
    model = {"past_mean": past_mean, "recent_mean": recent_mean, "past_loading": past_loading, "recent_loading": recent_loading}
    audit = {"rows": len(path), "past_dimension": 32, "recent_dimension": 32, "cca_rank": CCA_RANK, "cca_ridge": CCA_RIDGE, "canonical_correlations": singular[:CCA_RANK].tolist(), "past_loading_sha256": array_sha256(past_loading), "recent_loading_sha256": array_sha256(recent_loading), "past_covariance_eigen_min": float(x_eigen.min()), "recent_covariance_eigen_min": float(y_eigen.min()), "fixed_past_recent_split": True, "ridge_whitened_cross_covariance_svd": True, "label_free_train_only_cca": True, "model_fit_uses_target_rows": False, "target_labels_used": False}
    return model, audit


def cca_features(path, model):
    past = path[:, :, :4].reshape(len(path), 32)
    recent = path[:, :, 4:].reshape(len(path), 32)
    u = (past - model["past_mean"]) @ model["past_loading"]
    v = (recent - model["recent_mean"]) @ model["recent_loading"]
    feature = np.concatenate([u, v, u * v, v - u, np.abs(v - u)], axis=1)
    require(feature.shape == (len(path), FEATURE_DIMENSION) and np.isfinite(feature).all(), "V209 feature invalid")
    return feature


class RobustFeatureScale:
    def fit(self, feature):
        self.median = np.median(feature, axis=0)
        q25, q75 = np.quantile(feature, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(feature, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self

    def transform(self, feature):
        result = np.clip((feature - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V209 scaled feature invalid")
        return result


def cca_direction(train, target):
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, y, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    model, geometry = fit_cca(group_path)
    train_feature = cca_features(group_path, model)
    target_feature = cca_features(target_path, model)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    class_count = np.bincount(y, minlength=2).astype(float)
    require(class_count.min() > 0, "V209 class missing")
    sample_weight = np.where(y == 1, 0.5 / class_count[1], 0.5 / class_count[0])
    head = LogisticRegression(C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED)
    head.fit(train_design, y, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    geometry.update({"feature_dimension": FEATURE_DIMENSION, "feature_sha256": array_sha256(train_feature), "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": False, "chronological_slow_feature_generalized_eigen": False, "dmd_transition_operator": False, "pls_supervised_latent": False, "ica_independent_component": False})
    target_geometry = {**geometry, "rows": len(target_feature), "feature_sha256": array_sha256(target_feature), "model_fit_uses_target_rows": False}
    return probability, {"train_n": len(ordered), "train_event_group_n": len(y), "target_n": len(target), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit, "train_geometry": geometry, "target_geometry": target_geometry, "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": 7.0}, "feature_scaling": {"train_only": True, "median_sha256": array_sha256(feature_scale.median), "scale_sha256": array_sha256(feature_scale.scale), "clip": FEATURE_CLIP}, "head": {"type": "equal-group class-balanced ridge logistic", "C": RIDGE_LOGISTIC_C, "coefficient_sha256": array_sha256(head.coef_), "iterations": int(head.n_iter_[0])}, "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False, "model_fit_uses_target_rows": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False, "gpu_model_calls": 0, "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)}}


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base():
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = FEATURE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = cca_direction
    base.choose_inner = choose_inner


def choose_inner(train, validation, challenger):
    candidate, audit = cca_direction(train, validation)
    baseline = challenger.prob.to_numpy(float)
    confidence = challenger.confidence_signal.to_numpy(float)
    high = bool_series(challenger.high_conf).to_numpy(bool)
    baseline_metrics = metric(validation, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metrics, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2 * baseline_metrics["auc"] + baseline_metrics["balanced_accuracy"]}]
    geometry = audit["train_geometry"]
    model_ok = bool(geometry["fixed_past_recent_split"] and geometry["ridge_whitened_cross_covariance_svd"] and geometry["label_free_train_only_cca"] and not geometry["model_fit_uses_target_rows"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, candidate, architecture["weight"])
        metrics = metric(validation, probability, confidence, high)
        auc_delta = metrics["auc"] - baseline_metrics["auc"]
        ba_delta = metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
        net_delta = metrics["all_trade_mean_signed_net"] - baseline_metrics["all_trade_mean_signed_net"]
        source_floor, by_source = base.supported_source_ba_delta(validation, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": metrics, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": by_source, "model_eligible": model_ok, "eligible": bool(model_ok and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.01), "score": 2 * metrics["auc"] + metrics["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner OOF V69 noop or fixed .25/.50 cross-horizon regularized CCA blend", "baseline": baseline_metrics, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "split_rank_ridge_feature_head_blend_or_safety_micro_tuning": False}, audit


def run_nested(dev, challenger, smoke, smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):
        diagnostics, evidence, audits = base.run_nested(dev, challenger, smoke, smoke_market)
    diagnostics = diagnostics.rename(columns={"v146_model": "v209_model"})
    evidence = evidence.rename(columns={"frenet_prob": "cross_horizon_cca_prob", "v146_model": "v209_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V209 CROSS-HORIZON CCA] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}")
    return diagnostics, evidence, audits


def paired_nested_bootstrap(evidence, draws):
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V209 cross-horizon CCA policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(challenger, diagnostics, evidence, audits, draws, smoke):
    configure_base()
    base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(challenger, diagnostics, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = all(audit[side]["train_geometry"]["fixed_past_recent_split"] and audit[side]["train_geometry"]["ridge_whitened_cross_covariance_svd"] and audit[side]["train_geometry"]["label_free_train_only_cca"] and not audit[side]["model_fit_uses_target_rows"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_cross_horizon_regularized_cca_contract_verified"] = bool(contract)
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = challenger.copy()
        result["selected_summary"] = result["baseline_summary"]
        require(result["selected_frame"].equals(challenger), "V209 fallback")
    return result


def material_gate(evaluation):
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": sum(bool(value) for value in checks.values()), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key")
    return gate


def enforce_output_conflict_fail_closed(output):
    require(bool(CONTROLLER_OUTPUT) and output.resolve() == Path(CONTROLLER_OUTPUT).resolve() and output.resolve().parent == (ROOT / "research" / "staging" / "V209").resolve(), "controller binding")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in output.iterdir()}
    require(required.issubset(existing) and not (existing - allowed), "explicit conflict")
    return {"controller_bound": True}


def write_outputs(output, authority, evidence, audits, evaluation):
    conflict = enforce_output_conflict_fail_closed(output)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    reports = {
        "MODEL_COMPARISON.json": {"version": "V209", "models": {"V69": evaluation["baseline_summary"], "V209_DIAGNOSTIC": evaluation["candidate_summary"], "V209_SELECTED": selected}, "evaluation": compact, "authority": authority, "audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V209", "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V209", "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate},
        "V209_CROSS_HORIZON_CCA_REPORT.json": {"static_spec": STATIC_SPEC, "audits": audits, "evaluation": compact},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected)},
        "RUN_STATUS.json": {"version": VERSION, "status": evaluation["status"], "conflict": conflict},
    }
    for name, payload in reports.items():
        atomic_json(payload, output / name)
    atomic_csv(evidence, output / "V209_CROSS_HORIZON_CCA_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], output / "V209_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in output.iterdir() if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "files": files}, output / "ARTIFACT_MANIFEST.json")
    return {"artifact_count": len(files) + 1}


def parse_args():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not (arguments.audit_only or arguments.smoke_test or arguments.support_probe or arguments.full_run or arguments.output), "controller noarg conflict")
        arguments.full_run = True
        arguments.output = Path(CONTROLLER_OUTPUT)
    elif arguments.output is None:
        arguments.output = DEFAULT_OUT
    if arguments.full_run:
        require(bool(CONTROLLER_OUTPUT) and arguments.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "direct full")
    return arguments


def main():
    arguments = parse_args()
    bounded = bool(arguments.audit_only or arguments.smoke_test or arguments.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller deterministic CPU cross-horizon CCA", "gpu_model_calls": 0}
    if os.name == "nt":
        require(_kernel32.SetProcessAffinityMask(_process, ctypes.c_size_t(CPU_AFFINITY_MASK)), "V209 runtime affinity restore failed")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    runtime["observed_affinity"] = observed_affinity()
    runtime["gpu_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    runtime["gpu_model_calls"] = 0
    authority = verify_authority()
    dev, challenger = load_authorized()
    if arguments.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "static_spec": STATIC_SPEC, "canonical_controller_gate_checks": 14, "controller_contract": {"noarg": True, "conflict": True, "direct_full": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "nested_key": "selected.material_gate.nested_bootstrap", "native_robustness": True, "exact_fallback": True}, "output_written": False}), indent=2))
        return
    with threadpool_limits(limits=2):
        diagnostics, evidence, audits = run_nested(dev, challenger, bounded, arguments.smoke_market)
        if arguments.support_probe:
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"]]
            best = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"]))
            print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "runtime": runtime, "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "output_written": False}), indent=2))
            return
        evaluation = evaluate(challenger, diagnostics, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if arguments.smoke_test else BOOTSTRAP_DRAWS, arguments.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"]]
    best = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"]))
    summary = {"status": "SMOKE_OK" if arguments.smoke_test else evaluation["status"], "runtime": runtime, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if arguments.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]})
        print(json.dumps(clean(summary), indent=2))
        return
    print(json.dumps(clean({**summary, "output_written": True, "controller": write_outputs(arguments.output, authority, evidence, audits, evaluation)}), indent=2))


if __name__ == "__main__":
    main()
