"""V217 fixed 2D monogenic/Riesz phase-orientation direction challenger."""
from __future__ import annotations
import os
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_n] = "2"
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import experiment_v214_multichannel_causal_class_conditional_cramer_wold_ecf_direction as scaffold

core = scaffold.core
base = scaffold.base
ROOT = scaffold.ROOT
VERSION = 217
HYPOTHESIS = "MULTICHANNEL_CAUSAL_MONOGENIC_RIESZ_PHASE_DIRECTION_V1"
DEFAULT_OUT = ROOT / "research" / "staging" / "V217" / "LOCAL_FORBIDDEN"
FEATURES = scaffold.FEATURES
SCAFFOLD_PATH = ROOT / "experiment_v214_multichannel_causal_class_conditional_cramer_wold_ecf_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "2d51df6299de5a32fafe5234282f532b1d7d4500a63f047303a8368d1abb8280"
FEATURE_N = 240
REGION_N = 8
REGION_FEATURE_N = 30
SEED = 21701
ARCHITECTURES = (
    {"name": "MONOGENIC_RIESZ_PHASE_W0.25", "weight": 0.25},
    {"name": "MONOGENIC_RIESZ_PHASE_W0.50", "weight": 0.50},
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

def fixed_riesz_multipliers():
    frequency = np.fft.fftfreq(8)
    channel_frequency, horizon_frequency = np.meshgrid(frequency, frequency, indexing="ij")
    channel_symbol = np.sin(2.0 * np.pi * channel_frequency)
    horizon_symbol = np.sin(2.0 * np.pi * horizon_frequency)
    radius = np.sqrt(channel_symbol ** 2 + horizon_symbol ** 2)
    denominator = np.where(radius > 1e-12, radius, 1.0)
    first = -1j * channel_symbol / denominator
    second = -1j * horizon_symbol / denominator
    first[radius <= 1e-12] = 0.0
    second[radius <= 1e-12] = 0.0
    return first, second

def fixed_regions():
    row, col = np.indices((8, 8))
    masks = [
        np.ones((8, 8), dtype=bool),
        col < 4,
        col >= 4,
        row < 4,
        row >= 4,
        np.abs(row - col) <= 1,
        np.abs(row + col - 7) <= 1,
        (row >= 2) & (row < 6) & (col >= 2) & (col < 6),
    ]
    require(len(masks) == REGION_N and all(int(mask.sum()) >= 16 for mask in masks), "V217 regions")
    return tuple(masks)

RIESZ_CHANNEL, RIESZ_HORIZON = fixed_riesz_multipliers()
REGIONS = fixed_regions()
STATIC_SPEC = {
    "feature_dimension": FEATURE_N,
    "surface": "strict-train scaled continuous 8x8 causal channel-horizon surface",
    "transform": "fixed periodic central-difference 2D Riesz quadrature via exact FFT multipliers",
    "fields": ["monogenic amplitude", "local phase", "axial orientation"],
    "regions": REGION_N,
    "features_per_region": REGION_FEATURE_N,
    "head": "equal-event-group class-balanced C=.5 ridge logistic",
    "target_model_fit": False,
}

class RobustFeatureScale:
    def __init__(self):
        self.median = np.zeros(FEATURE_N)
        self.scale = np.ones(FEATURE_N)
    def fit(self, matrix):
        require(matrix.ndim == 2 and matrix.shape[1] == FEATURE_N, "V217 feature fit shape")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        return self
    def transform(self, matrix):
        require(matrix.ndim == 2 and matrix.shape[1] == FEATURE_N, "V217 feature transform shape")
        return np.clip((matrix - self.median) / self.scale, -7.0, 7.0)

def verify_authority():
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "V214 scaffold changed")
    authority = scaffold.verify_authority()
    authority["code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "authority/chronology/path/metrics/gate/bootstrap/IO only; no V214 output"}
    authority["v217_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    return authority

def _region_summary(surface, first, second, amplitude, phase, orientation, mask):
    value = surface[:, mask]
    r1 = first[:, mask]
    r2 = second[:, mask]
    amp = amplitude[:, mask]
    phi = phase[:, mask]
    theta = orientation[:, mask]
    mean = lambda x: np.mean(x, axis=1)
    std = lambda x: np.std(x, axis=1)
    quantile = lambda x, q: np.quantile(x, q, axis=1)
    eps = 1e-8
    amp_weight = amp / np.maximum(mean(amp)[:, None], eps)
    columns = [
        mean(value), std(value), quantile(value, 0.25), quantile(value, 0.75), mean(np.abs(value)),
        mean(amp), std(amp), quantile(amp, 0.25), quantile(amp, 0.75), np.max(amp, axis=1),
        mean(np.cos(phi)), mean(np.sin(phi)), np.hypot(mean(np.cos(phi)), mean(np.sin(phi))), mean(np.cos(2 * phi)), mean(np.sin(2 * phi)),
        mean(np.cos(2 * theta)), mean(np.sin(2 * theta)), np.hypot(mean(np.cos(2 * theta)), mean(np.sin(2 * theta))), mean(np.cos(4 * theta)), mean(np.sin(4 * theta)),
        mean(amp_weight * np.cos(phi)), mean(amp_weight * np.sin(phi)), mean(amp_weight * np.cos(2 * theta)), mean(amp_weight * np.sin(2 * theta)),
        mean(r1), std(r1), mean(r2), std(r2), mean(np.cos(phi - 2 * theta)), mean(np.sin(phi - 2 * theta)),
    ]
    return np.stack(columns, axis=1)

def monogenic_features(path):
    require(path.ndim == 3 and path.shape[1:] == (8, 8), "V217 path shape")
    spectrum = np.fft.fft2(path, axes=(1, 2))
    first_complex = np.fft.ifft2(spectrum * RIESZ_CHANNEL[None, :, :], axes=(1, 2))
    second_complex = np.fft.ifft2(spectrum * RIESZ_HORIZON[None, :, :], axes=(1, 2))
    imaginary_leak = max(float(np.max(np.abs(first_complex.imag))), float(np.max(np.abs(second_complex.imag))))
    require(imaginary_leak < 1e-9, "V217 Riesz conjugate symmetry")
    first = first_complex.real
    second = second_complex.real
    quadrature = np.sqrt(first ** 2 + second ** 2)
    amplitude = np.sqrt(path ** 2 + quadrature ** 2)
    phase = np.arctan2(quadrature, path)
    orientation = np.arctan2(second, first)
    feature = np.concatenate([_region_summary(path, first, second, amplitude, phase, orientation, mask) for mask in REGIONS], axis=1)
    require(feature.shape == (len(path), FEATURE_N) and np.isfinite(feature).all(), "V217 monogenic feature")
    audit = {
        "rows": len(path), "feature_dimension": FEATURE_N, "fixed_discrete_riesz_multipliers": True,
        "two_quadrature_fields": True, "monogenic_amplitude_phase_orientation": True,
        "regional_circular_and_axial_moments": True, "label_independent_exact_transform": True,
        "local_differential_geometry_not_global_path_integral": True,
        "imaginary_leak_max": imaginary_leak, "riesz_channel_sha256": array_sha256(RIESZ_CHANNEL),
        "riesz_horizon_sha256": array_sha256(RIESZ_HORIZON), "model_fit_uses_target_rows": False,
        "target_labels_used": False, "v160_one_dimensional_hilbert_per_channel": False,
        "v163_frequency_triad_bispectrum": False, "v179_gradient_structure_tensor": False,
        "v186_homomorphic_cepstrum": False, "v214_class_ecf_prototype": False,
        "v215_population_diffusion_map": False, "v216_class_gmrf_precision": False,
    }
    return feature, audit

def monogenic_direction(train, target):
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    train_feature, train_geometry = monogenic_features(group_path)
    target_feature, target_geometry = monogenic_features(target_path)
    feature_scale = RobustFeatureScale().fit(train_feature)
    train_design = feature_scale.transform(train_feature)
    target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    require(np.all(class_n > 0), "V217 class support")
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(C=0.5, penalty="l2", solver="liblinear", max_iter=300, random_state=SEED)
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1 - 1e-6)
    return probability, {
        "train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target),
        "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping,
        "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit,
        "train_geometry": train_geometry, "target_geometry": target_geometry,
        "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": 7.0},
        "feature_scaling": {"train_only": True, "median_sha256": array_sha256(feature_scale.median), "scale_sha256": array_sha256(feature_scale.scale), "clip": 7.0},
        "head": {"type": "fixed equal-event-group/class-balanced ridge logistic", "C": 0.5, "solver": "liblinear", "coefficient_sha256": array_sha256(head.coef_), "coefficient_l2": float(np.linalg.norm(head.coef_)), "intercept": head.intercept_.tolist()},
        "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False,
        "model_fit_uses_target_rows": False,
        "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False,
        "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False,
        "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False,
        "source_time_environment_transport_or_projection": False,
        "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)},
    }

_BASE_BOOTSTRAP = scaffold._BASE_BOOTSTRAP
def configure_base():
    core.VERSION = VERSION; core.HYPOTHESIS = HYPOTHESIS; core.STATIC_SPEC = STATIC_SPEC; core.DAG_FEATURE_DIMENSION = FEATURE_N; core.ARCHITECTURES = ARCHITECTURES; core.SEED = SEED; core.dag_direction = monogenic_direction; core.choose_inner = choose_inner
    base.VERSION = VERSION; base.HYPOTHESIS = HYPOTHESIS; base.STATIC_SPEC = STATIC_SPEC; base.FRENET_FEATURE_DIMENSION = FEATURE_N; base.ARCHITECTURES = ARCHITECTURES; base.SEED = SEED; base.frenet_direction = monogenic_direction

def choose_inner(train, valid, champion):
    challenger, audit = monogenic_direction(train, valid)
    baseline = champion.prob.to_numpy(float); confidence = champion.confidence_signal.to_numpy(float); high = bool_series(champion.high_conf).to_numpy(bool)
    baseline_metric = metric(valid, baseline, confidence, high)
    trials = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    geometry = audit["train_geometry"]
    model_ok = bool(geometry["fixed_discrete_riesz_multipliers"] and geometry["two_quadrature_fields"] and geometry["monogenic_amplitude_phase_orientation"] and geometry["regional_circular_and_axial_moments"] and not geometry["model_fit_uses_target_rows"] and not audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        measured = metric(valid, probability, confidence, high)
        auc_delta = measured["auc"] - baseline_metric["auc"]
        ba_delta = measured["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = measured["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_delta = base.supported_source_ba_delta(valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": measured, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_delta, "model_eligible": model_ok, "eligible": bool(model_ok and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.01), "score": 2 * measured["auc"] + measured["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner OOF V69 noop or fixed .25/.50 monogenic Riesz phase blend", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "multiplier_region_summary_head_blend_or_safety_micro_tuning": False}, audit

def run_nested(dev, champion, smoke, smoke_market="US"):
    configure_base()
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = core.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v195_model": "v217_model"})
    evidence = evidence.rename(columns={"sparse_dag_moment_prob": "monogenic_riesz_prob", "v195_model": "v217_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V217 MONOGENIC] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}")
    return diagnostic, evidence, audits

def paired_nested_bootstrap(evidence, draws):
    configure_base(); result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V217 monogenic Riesz policy"; result["seed"] = SEED; result["standardized"] = True
    return result

def evaluate(champion, diagnostic, evidence, audits, draws, smoke):
    configure_base(); base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]; checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = all(audit[side]["train_geometry"]["fixed_discrete_riesz_multipliers"] and audit[side]["train_geometry"]["monogenic_amplitude_phase_orientation"] and audit[side]["train_geometry"]["regional_circular_and_axial_moments"] and not audit[side]["train_geometry"]["model_fit_uses_target_rows"] and not audit[side]["target_labels_used"] for audit in audits for side in ("inner_model_audit", "outer_model_audit"))
    checks["multichannel_causal_monogenic_riesz_phase_contract_verified"] = bool(contract)
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy(); result["selected_summary"] = result["baseline_summary"]
        require(result["selected_frame"].equals(champion), "V217 exact fallback")
    return result

def material_gate(evaluation):
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": sum(bool(value) for value in checks.values()), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V217 nested key")
    return gate

def enforce_output_conflict_fail_closed(out):
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve() and out.resolve().parent == (ROOT / "research" / "staging" / "V217").resolve(), "V217 binding")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}; allowed = required | {"DATA_EPOCH_BINDING.json"}; existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing) and not (existing - allowed), "V217 conflict")
    return {"controller_bound": True}

def write_outputs(out, authority, evidence, audits, evaluation):
    conflict = enforce_output_conflict_fail_closed(out); compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}; gate = material_gate(evaluation); selected = json.loads(json.dumps(clean(evaluation["selected_summary"]))); selected["material_gate"] = gate
    reports = {
        "MODEL_COMPARISON.json": {"version": "V217", "status": evaluation["status"], "models": {"V69": evaluation["baseline_summary"], "V217_DIAGNOSTIC": evaluation["candidate_summary"], "V217_SELECTED": selected}, "evaluation": compact, "authority": authority, "audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V217", "status": evaluation["status"], "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V217", "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate},
        "V217_MONOGENIC_RIESZ_PHASE_REPORT.json": {"status": evaluation["status"], "static_spec": STATIC_SPEC, "audits": audits, "evaluation": compact},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected)},
        "RUN_STATUS.json": {"version": VERSION, "status": evaluation["status"], "priority": 875, "conflict": conflict},
    }
    for name, payload in reports.items(): atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V217_MONOGENIC_RIESZ_EVIDENCE.csv.gz"); atomic_csv(evaluation["selected_frame"], out / "V217_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in out.iterdir() if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}; atomic_json({"version": VERSION, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"artifact_count": len(files) + 1}

def parse_args():
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT)); mode.add_argument("--audit-only", action="store_true"); mode.add_argument("--smoke-test", action="store_true"); mode.add_argument("--support-probe", action="store_true"); mode.add_argument("--full-run", action="store_true"); parser.add_argument("--smoke-market", choices=("US", "KR"), default="US"); parser.add_argument("--output", type=Path); args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not (args.audit_only or args.smoke_test or args.support_probe or args.full_run or args.output), "V217 controller conflict"); args.full_run = True; args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None: args.output = DEFAULT_OUT
    if args.full_run: require(bool(CONTROLLER_OUTPUT) and args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V217 direct full")
    return args

def main():
    args = parse_args(); bounded = bool(args.audit_only or args.smoke_test or args.support_probe); runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller deterministic CPU monogenic Riesz phase", "gpu_model_calls": 0}; authority = verify_authority(); dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "static_spec": STATIC_SPEC, "canonical_controller_gate_checks": 14, "controller_contract": {"noarg": True, "conflict": True, "direct_full": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "dev_robustness_top_level_status": True, "nested_key": "selected.material_gate.nested_bootstrap", "exact_fallback": True}, "output_written": False}), indent=2)); return
    with threadpool_limits(limits=2 if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, bounded, args.smoke_market)
        if args.support_probe:
            audit = audits[0]; non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"]]; best = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"])); print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "runtime": runtime, "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "output_written": False}), indent=2)); return
        evaluation = evaluate(champion, diagnostic, evidence, audits, 250 if args.smoke_test else 2000, args.smoke_test)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"]]; best = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"])); summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "runtime": runtime, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]]}); print(json.dumps(clean(summary), indent=2)); return
    print(json.dumps(clean({**summary, "output_written": True, "controller": write_outputs(args.output, authority, evidence, audits, evaluation)}), indent=2))

if __name__ == "__main__":
    main()
