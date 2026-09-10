"""V204 normalized two-additive Choquet-capacity causal direction."""
from __future__ import annotations
import os
import ctypes

# Every V204 process is pinned before numerical-library imports.  This is also
# deliberately retained for controller execution because the model is CPU-only.
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
        raise RuntimeError("V204 could not pin CPU30-31 before numerical imports")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import experiment_v202_multichannel_causal_kernel_stein_discrepancy_direction as scaffold
# A legacy scaffold import binds its controller default to CUDA0.  V204 is
# CPU-only, so restore both runtime contracts immediately after import.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if os.name == "nt" and not _kernel32.SetProcessAffinityMask(_process, ctypes.c_size_t(CPU_AFFINITY_MASK)):
    raise RuntimeError("V204 could not restore CPU30-31 after scaffold import")

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
core = scaffold.scaffold
base = scaffold.base
ROOT = scaffold.ROOT
VERSION = 204
HYPOTHESIS = "MULTICHANNEL_CAUSAL_TWO_ADDITIVE_CHOQUET_CAPACITY_DIRECTION_V1"
DEFAULT_OUT = ROOT / "research" / "staging" / "V204" / "LOCAL_FORBIDDEN"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v202_multichannel_causal_kernel_stein_discrepancy_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "977cea12302bc095183c3ccbaf3be31749176b1f79e160158192715275526b05"

CRITERION_DIM = 16
PAIR_DIM = CRITERION_DIM * (CRITERION_DIM - 1) // 2
FEATURE_DIMENSION = CRITERION_DIM + PAIR_DIM
CAPACITY_L2 = 0.02
OPTIMIZER_MAX_ITER = 600
SEED = 20401
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
ARCHITECTURES = (
    {"name": "TWO_ADDITIVE_CHOQUET_W0.25", "weight": 0.25},
    {"name": "TWO_ADDITIVE_CHOQUET_W0.50", "weight": 0.50},
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
    "criteria": "eight semantic-channel horizon medians, pooled strict-train ECDF values and complements",
    "criterion_dimension": CRITERION_DIM,
    "basis": "all singleton criteria and pairwise minima",
    "feature_dimension": FEATURE_DIMENSION,
    "capacity": "normalized monotone two-additive fuzzy capacity with nonnegative simplex Mobius masses",
    "head": "event-group/class-balanced logistic Choquet score with fixed L2 and analytic-gradient L-BFGS-B",
    "capacity_l2": CAPACITY_L2,
    "optimizer_max_iter": OPTIMIZER_MAX_ITER,
    "target_model_fit": False,
    "gpu_model_calls": 0,
}


def observed_affinity():
    if os.name != "nt":
        return {"mask_hex": None, "cpus": None}
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    ok = _kernel32.GetProcessAffinityMask(_process, ctypes.byref(process_mask), ctypes.byref(system_mask))
    require(bool(ok), "V204 GetProcessAffinityMask failed")
    cpus = [i for i in range(64) if process_mask.value & (1 << i)]
    require(process_mask.value == CPU_AFFINITY_MASK and cpus == [30, 31], "V204 affinity drift")
    return {"mask_hex": f"0x{process_mask.value:08X}", "cpus": cpus}


def verify_authority():
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "V202 scaffold changed")
    authority = scaffold.verify_authority()
    authority["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "authority/chronology/path/metrics/gate/bootstrap/IO only; no V202 output",
    }
    authority["v204_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
        "prior_version_output_read": False,
    }
    return authority


def empirical_criteria(train_surface, target_surface):
    train_summary = np.median(train_surface, axis=2)
    target_summary = np.median(target_surface, axis=2)
    n = len(train_summary)
    require(n >= 40 and train_summary.shape[1] == 8, "V204 ECDF support")
    train_q = np.empty_like(train_summary, dtype=float)
    target_q = np.empty_like(target_summary, dtype=float)
    cut_sha = []
    for j in range(8):
        cut = np.sort(train_summary[:, j], kind="mergesort")
        train_q[:, j] = (np.searchsorted(cut, train_summary[:, j], side="left") + np.searchsorted(cut, train_summary[:, j], side="right")) / (2.0 * n)
        target_q[:, j] = (np.searchsorted(cut, target_summary[:, j], side="left") + np.searchsorted(cut, target_summary[:, j], side="right")) / (2.0 * n)
        cut_sha.append(array_sha256(cut))
    train_q = np.clip(train_q, 1.0 / (2.0 * n), 1.0 - 1.0 / (2.0 * n))
    target_q = np.clip(target_q, 1.0 / (2.0 * n), 1.0 - 1.0 / (2.0 * n))
    train_criteria = np.concatenate([train_q, 1.0 - train_q], axis=1)
    target_criteria = np.concatenate([target_q, 1.0 - target_q], axis=1)
    return train_criteria, target_criteria, cut_sha


def choquet_basis(criteria):
    require(criteria.ndim == 2 and criteria.shape[1] == CRITERION_DIM, "V204 criterion dimension")
    columns = [criteria]
    pair_index = []
    pair_columns = []
    for i in range(CRITERION_DIM):
        for j in range(i + 1, CRITERION_DIM):
            pair_columns.append(np.minimum(criteria[:, i], criteria[:, j]))
            pair_index.append((i, j))
    columns.append(np.column_stack(pair_columns))
    basis = np.concatenate(columns, axis=1)
    require(basis.shape[1] == FEATURE_DIMENSION and np.isfinite(basis).all(), "V204 basis invalid")
    return basis, pair_index


def fit_capacity(train_basis, y):
    y = np.asarray(y, dtype=float)
    class_count = np.bincount(y.astype(int), minlength=2).astype(float)
    require(class_count.min() >= 15, "V204 class support")
    sample_weight = 0.5 / class_count[y.astype(int)]
    sample_weight /= sample_weight.sum()
    dimension = train_basis.shape[1]
    # A nonnegative coefficient vector is exactly a positive scale times a
    # normalized capacity mass vector.  Fitting that vector directly makes the
    # Bernoulli objective convex; normalization is applied only for the stored
    # Choquet capacity and cannot alter the fitted logit.
    initial = np.concatenate([np.full(dimension, 2.0 / dimension), [0.0]])

    def objective(parameter):
        positive_coefficient = parameter[:dimension]
        bias = parameter[-1]
        logit = bias + train_basis @ positive_coefficient
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0)))
        loss = -np.sum(sample_weight * (y * np.log(probability + 1e-12) + (1.0 - y) * np.log(1.0 - probability + 1e-12)))
        loss += CAPACITY_L2 * float(np.dot(positive_coefficient, positive_coefficient))
        gradient_logit = sample_weight * (probability - y)
        gradient_coefficient = train_basis.T @ gradient_logit + 2.0 * CAPACITY_L2 * positive_coefficient
        gradient_bias = float(gradient_logit.sum())
        gradient = np.concatenate([gradient_coefficient, [gradient_bias]])
        return float(loss), gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * dimension + [(None, None)],
        options={"maxiter": OPTIMIZER_MAX_ITER, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    )
    require(np.isfinite(result.fun) and np.isfinite(result.x).all(), "V204 optimizer nonfinite")
    require(bool(result.success) or float(np.linalg.norm(result.jac, ord=np.inf)) <= 5e-5, f"V204 optimizer failed: {result.message}")
    positive_coefficient = np.maximum(result.x[:dimension], 0.0)
    scale = float(positive_coefficient.sum())
    require(scale > 1e-8, "V204 degenerate capacity scale")
    masses = positive_coefficient / scale
    return masses, float(result.x[-1]), scale, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "gradient_inf_norm": float(np.linalg.norm(result.jac, ord=np.inf)),
    }


def choquet_direction(train, target):
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_audit = processor.transform(ordered)
    target_state, target_audit = processor.transform(target)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, y, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    train_surface = channel_scale.transform(group_path)
    target_surface = channel_scale.transform(target_path)
    train_criteria, target_criteria, ecdf_sha = empirical_criteria(train_surface, target_surface)
    train_basis, pair_index = choquet_basis(train_criteria)
    target_basis, _ = choquet_basis(target_criteria)
    masses, bias, scale, optimizer = fit_capacity(train_basis, y)
    target_choquet = target_basis @ masses
    logit = bias + scale * target_choquet
    probability = np.clip(1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0))), 1e-6, 1.0 - 1e-6)
    geometry = {
        "rows": len(train_basis),
        "criterion_dimension": CRITERION_DIM,
        "feature_dimension": FEATURE_DIMENSION,
        "singleton_mass_sum": float(masses[:CRITERION_DIM].sum()),
        "pair_mass_sum": float(masses[CRITERION_DIM:].sum()),
        "capacity_mass_sum": float(masses.sum()),
        "capacity_mass_min": float(masses.min()),
        "capacity_mass_sha256": array_sha256(masses),
        "pair_index_sha256": array_sha256(np.asarray(pair_index, dtype=np.int64)),
        "ecdf_cut_sha256": ecdf_sha,
        "optimizer": optimizer,
        "normalized_two_additive_monotone_capacity": True,
        "singleton_and_pairwise_minimum_basis": True,
        "pooled_strict_train_ecdf_and_complements": True,
        "event_group_class_balanced_supervised_head": True,
        "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": False,
        "model_fit_uses_target_rows": False,
        "target_labels_used": False,
        "choquet_or_fuzzy_capacity": True,
        "potts_grid_or_pseudolikelihood": False,
        "rasch_item_response_or_latent_ability": False,
    }
    return probability, {
        "train_n": len(ordered),
        "train_event_group_n": len(y),
        "target_n": len(target),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_audit,
        "target_transform": target_audit,
        "grouping": grouping,
        "static_spec": STATIC_SPEC,
        "train_path": train_path_audit,
        "target_path": target_path_audit,
        "train_geometry": geometry,
        "target_geometry": {**geometry, "rows": len(target_basis), "model_fit_uses_target_rows": False},
        "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": 7.0},
        "feature_scaling": {"train_only": True, "pooled_ecdf": True, "target_batch_fit": False},
        "head": {"type": "normalized two-additive Choquet logistic capacity", "bias": bias, "scale": scale, "l2": CAPACITY_L2},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False,
        "target_labels_used": False,
        "model_fit_uses_target_rows": False,
        "haar_wavelet_or_scattering": False,
        "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False,
        "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False,
        "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "kernel_nystrom_or_random_features": False,
        "class_density_or_likelihood_ratio": False,
        "grid_texture_or_graph_model": False,
        "source_time_environment_model": False,
        "gpu_model_calls": 0,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }


_BASE_BOOTSTRAP = scaffold._BASE_BOOTSTRAP


def configure_base():
    core.VERSION = VERSION
    core.HYPOTHESIS = HYPOTHESIS
    core.STATIC_SPEC = STATIC_SPEC
    core.DAG_FEATURE_DIMENSION = FEATURE_DIMENSION
    core.ARCHITECTURES = ARCHITECTURES
    core.SEED = SEED
    core.dag_direction = choquet_direction
    core.choose_inner = choose_inner
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = FEATURE_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = choquet_direction
    base.choose_inner = choose_inner


def choose_inner(train, validation, challenger):
    candidate, audit = choquet_direction(train, validation)
    baseline = challenger.prob.to_numpy(float)
    confidence = challenger.confidence_signal.to_numpy(float)
    high = bool_series(challenger.high_conf).to_numpy(bool)
    baseline_metrics = metric(validation, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metrics, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2 * baseline_metrics["auc"] + baseline_metrics["balanced_accuracy"]}]
    geometry = audit["train_geometry"]
    model_ok = bool(geometry["normalized_two_additive_monotone_capacity"] and geometry["singleton_and_pairwise_minimum_basis"] and geometry["pooled_strict_train_ecdf_and_complements"] and abs(geometry["capacity_mass_sum"] - 1.0) < 1e-10 and geometry["capacity_mass_min"] >= 0.0 and not geometry["model_fit_uses_target_rows"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, candidate, architecture["weight"])
        metrics = metric(validation, probability, confidence, high)
        auc_delta = metrics["auc"] - baseline_metrics["auc"]
        ba_delta = metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
        net_delta = metrics["all_trade_mean_signed_net"] - baseline_metrics["all_trade_mean_signed_net"]
        source_floor, by_source = base.supported_source_ba_delta(validation, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": metrics, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": by_source, "model_eligible": model_ok, "eligible": bool(model_ok and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.01), "score": 2 * metrics["auc"] + metrics["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner OOF V69 noop or fixed .25/.50 normalized two-additive Choquet-capacity blend", "baseline": baseline_metrics, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "ecdf_capacity_basis_optimizer_blend_or_safety_micro_tuning": False}, audit


def run_nested(dev, challenger, smoke, smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):
        diagnostics, evidence, audits = base.run_nested(dev, challenger, smoke, smoke_market)
    diagnostics = diagnostics.rename(columns={"v146_model": "v204_model"})
    evidence = evidence.rename(columns={"frenet_prob": "choquet_capacity_prob", "v146_model": "v204_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V204 CHOQUET] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}")
    return diagnostics, evidence, audits


def paired_nested_bootstrap(evidence, draws):
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V204 Choquet-capacity policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(challenger, diagnostics, evidence, audits, draws, smoke):
    configure_base()
    base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(challenger, diagnostics, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = all(audit[side]["train_geometry"]["normalized_two_additive_monotone_capacity"] and audit[side]["train_geometry"]["singleton_and_pairwise_minimum_basis"] and audit[side]["train_geometry"]["pooled_strict_train_ecdf_and_complements"] and not audit[side]["model_fit_uses_target_rows"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_two_additive_choquet_capacity_contract_verified"] = bool(contract)
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = challenger.copy()
        result["selected_summary"] = result["baseline_summary"]
        require(result["selected_frame"].equals(challenger), "V204 fallback")
    return result


def material_gate(evaluation):
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": sum(bool(value) for value in checks.values()), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key")
    return gate


def enforce_output_conflict_fail_closed(output):
    require(bool(CONTROLLER_OUTPUT) and output.resolve() == Path(CONTROLLER_OUTPUT).resolve() and output.resolve().parent == (ROOT / "research" / "staging" / "V204").resolve(), "controller binding")
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
        "MODEL_COMPARISON.json": {"version": "V204", "models": {"V69": evaluation["baseline_summary"], "V204_DIAGNOSTIC": evaluation["candidate_summary"], "V204_SELECTED": selected}, "evaluation": compact, "authority": authority, "audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V204", "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V204", "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate},
        "V204_CHOQUET_CAPACITY_REPORT.json": {"static_spec": STATIC_SPEC, "audits": audits, "evaluation": compact},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected)},
        "RUN_STATUS.json": {"version": VERSION, "status": evaluation["status"], "conflict": conflict},
    }
    for name, payload in reports.items():
        atomic_json(payload, output / name)
    atomic_csv(evidence, output / "V204_CHOQUET_CAPACITY_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], output / "V204_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    return arguments


def main():
    arguments = parse_args()
    bounded = bool(arguments.audit_only or arguments.smoke_test or arguments.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller deterministic CPU Choquet capacity", "gpu_model_calls": 0}
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
