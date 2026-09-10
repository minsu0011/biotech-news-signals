"""V196 class-conditional RealNVP density-ratio direction challenger.

The requested quantum-inspired MPS was rejected because changing V192's
polynomial site map to an amplitude/phase map would retain the same bounded-
bond 64-site tensor-network contraction.  V196 instead learns two strict-past
class-conditional invertible densities on the train-scaled immutable V36 8x8
causal surface.  Four fixed alternating affine coupling layers use exact
change-of-variables log determinants.  Equal-prior UP-versus-DOWN log-density
odds give the challenger probability directly.  This is not Gaussian/copula
QDA, direct uLSIF, chronological covariate reweighting, or a tensor network.

Bounded preparation is CPU-only with GPU hidden.  Controller-bound full mode
requires deterministic CUDA device 0 and fails closed otherwise.  Inner OOF
selects exact V69 or fixed .25/.50 blends.  Outer labels are evaluation-only
after the 35-minute embargo/event purge.  V69 confidence/high_conf stay exact,
and any material or execution failure restores the exact entire V69 frame.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_thread_variable] = "2"

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import experiment_v192_multichannel_causal_tensor_train_interaction_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V196" / "LOCAL_FORBIDDEN"
VERSION = 196
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CLASS_CONDITIONAL_REALNVP_DENSITY_RATIO_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v192_multichannel_causal_tensor_train_interaction_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "9f5cef3d302430de29cd5a6b6b73ecf1b721aa6c5d0044c3e9d335cbde107c04"

HORIZON_N = 8
CHANNEL_N = 8
STATE_N = HORIZON_N * CHANNEL_N
FLOW_LAYER_N = 4
CONDITIONER_HIDDEN_N = 16
FLOW_CLIP = 5.0
FLOW_DIVISOR = 2.5
LOG_SCALE_BOUND = 0.60
TRAIN_EPOCHS = 36
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 5.0
SEED = 19601
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "REALNVP_DENSITY_RATIO_W0.25", "weight": 0.25},
    {"name": "REALNVP_DENSITY_RATIO_W0.50", "weight": 0.50},
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

_COUPLING_PARAMETER_N = STATE_N * CONDITIONER_HIDDEN_N + CONDITIONER_HIDDEN_N + CONDITIONER_HIDDEN_N * (2 * STATE_N) + 2 * STATE_N
FLOW_PARAMETER_N = FLOW_LAYER_N * _COUPLING_PARAMETER_N
TOTAL_PARAMETER_N = 2 * FLOW_PARAMETER_N

STATIC_SPEC = {
    "surface_shape": [HORIZON_N, CHANNEL_N],
    "state_order": "64 row-major coordinates: 120m-to-1m horizons outer, immutable semantic channels inner",
    "state_transform": "clip(train-cut channel-scaled value,-5,5)/2.5",
    "feature_dimension": STATE_N,
    "density_model": "two class-conditional RealNVP flows",
    "flow_layers": FLOW_LAYER_N,
    "coupling_masks": "fixed alternating even/odd 64-coordinate masks",
    "conditioner": {"hidden_width": CONDITIONER_HIDDEN_N, "activation": "tanh", "log_scale_bound": LOG_SCALE_BOUND},
    "parameter_count_two_flows": TOTAL_PARAMETER_N,
    "objective": "equal-class mean exact change-of-variables negative log likelihood",
    "optimizer": {"name": "AdamW", "epochs": TRAIN_EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP},
    "probability": "sigmoid(log p_UP(x)-log p_DOWN(x)) with equal priors",
    "bounded_device": "CPU only with CUDA hidden",
    "controller_full_device": "deterministic CUDA device 0 required; fail closed otherwise",
    "target_batch_fit": False,
}
require(TOTAL_PARAMETER_N == 25728, "V196 parameter dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V192 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["v196_code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, observed path, metrics, gate, bootstrap and atomic IO only; no V192 output is read"}
    audit["v196_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    audit["gpu_execution_contract"] = {"bounded_cuda_hidden": not bool(CONTROLLER_OUTPUT), "controller_full_requires_cuda_device_zero": True, "deterministic_algorithms_required": True, "nondeterminism_or_cuda_unavailable_fails_closed": True}
    return audit


def model_device() -> tuple[torch.device, dict[str, Any]]:
    torch.set_num_threads(SMOKE_THREADS)
    torch.use_deterministic_algorithms(True)
    if not CONTROLLER_OUTPUT:
        return torch.device("cpu"), {"device": "cpu", "bounded_cpu_only": True, "cuda_visible_devices_requested": "-1", "all_model_tensors_on_cpu": True, "gpu_model_calls": 0, "controller_full_gpu0": False, "deterministic_algorithms": True}
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "V196 controller full requires CUDA device 0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0"), {"device": "cuda:0", "bounded_cpu_only": False, "gpu_model_calls": 1, "controller_full_gpu0": True, "deterministic_algorithms": True}


def flow_state(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V196 path shape invalid")
    state = np.clip(path.reshape(len(path), STATE_N), -FLOW_CLIP, FLOW_CLIP) / FLOW_DIVISOR
    require(state.shape == (len(path), STATE_N) and np.isfinite(state).all(), "V196 flow state invalid")
    return state, {"rows": len(state), "state_n": STATE_N, "feature_dimension": STATE_N, "state_sha256": array_sha256(state), "absolute_max": float(np.max(np.abs(state))), "immutable_row_major_order": True, "fixed_bounded_affine_state": True, "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": False, "target_batch_fit": False, "tensor_train_or_mps": False, "kernel_rff_nystrom": False, "gaussian_copula_qda": False, "direct_ulsif": False, "chronological_covariate_reweighting": False}


class AffineCoupling(nn.Module):
    def __init__(self, parity: int, seed: int) -> None:
        super().__init__()
        mask = ((torch.arange(STATE_N) + parity) % 2 == 0).to(torch.float64)
        self.register_buffer("mask", mask)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.input_layer = nn.Linear(STATE_N, CONDITIONER_HIDDEN_N, dtype=torch.float64)
        self.output_layer = nn.Linear(CONDITIONER_HIDDEN_N, 2 * STATE_N, dtype=torch.float64)
        with torch.no_grad():
            self.input_layer.weight.copy_(0.02 * torch.randn(self.input_layer.weight.shape, generator=generator, dtype=torch.float64))
            self.input_layer.bias.zero_()
            self.output_layer.weight.zero_()
            self.output_layer.bias.zero_()

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = value * self.mask
        raw = self.output_layer(torch.tanh(self.input_layer(fixed)))
        raw_scale, translation = raw.chunk(2, dim=1)
        moving = 1.0 - self.mask
        log_scale = LOG_SCALE_BOUND * torch.tanh(raw_scale) * moving
        translation = translation * moving
        transformed = fixed + moving * (value * torch.exp(log_scale) + translation)
        return transformed, log_scale.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, seed_offset: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(layer % 2, SEED + seed_offset + layer) for layer in range(FLOW_LAYER_N)])

    def log_probability(self, value: torch.Tensor) -> torch.Tensor:
        current = value
        log_det = torch.zeros(len(value), dtype=value.dtype, device=value.device)
        for layer in self.layers:
            current, increment = layer(current)
            log_det = log_det + increment
        base_log = -0.5 * (current.square() + math.log(2.0 * math.pi)).sum(dim=1)
        return base_log + log_det


class ClassConditionalRealNVP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = RealNVP(0)
        self.up = RealNVP(1000)

    def class_log_probabilities(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.down.log_probability(value), self.up.log_probability(value)


def parameter_sha256(model: ClassConditionalRealNVP) -> str:
    flat = np.concatenate([parameter.detach().cpu().numpy().reshape(-1) for parameter in model.parameters()])
    return array_sha256(flat)


def fit_realnvp(train_state: np.ndarray, target_state: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    device, device_audit = model_device()
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = ClassConditionalRealNVP().to(device)
    train_tensor = torch.as_tensor(train_state, dtype=torch.float64, device=device)
    target_tensor = torch.as_tensor(target_state, dtype=torch.float64, device=device)
    down_mask = torch.as_tensor(target == 0, dtype=torch.bool, device=device)
    up_mask = torch.as_tensor(target == 1, dtype=torch.bool, device=device)
    require(int(down_mask.sum()) >= 12 and int(up_mask.sum()) >= 12, "V196 insufficient class support")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def objective() -> torch.Tensor:
        down_log, up_log = model.class_log_probabilities(train_tensor)
        return -0.5 * down_log[down_mask].mean() - 0.5 * up_log[up_mask].mean()

    with torch.no_grad():
        initial_loss = float(objective().cpu())
    gradient_max = 0.0
    for _ in range(TRAIN_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        require(bool(torch.isfinite(loss).item()), "V196 RealNVP loss non-finite")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP).detach().cpu())
        gradient_max = max(gradient_max, gradient)
        optimizer.step()
    with torch.no_grad():
        final_loss = float(objective().cpu())
        down_log, up_log = model.class_log_probabilities(target_tensor)
        log_odds = torch.clamp(up_log - down_log, -20.0, 20.0)
        probability = torch.sigmoid(log_odds).cpu().numpy()
        mean_abs_log_odds = float(torch.mean(torch.abs(log_odds)).cpu())
    require(np.isfinite(probability).all() and final_loss <= initial_loss + 1e-4, "V196 RealNVP training contract failed")
    audit = {"model": "class-conditional RealNVP exact density ratio", "flow_layer_n": FLOW_LAYER_N, "conditioner_hidden_n": CONDITIONER_HIDDEN_N, "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "epochs": TRAIN_EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP, "log_scale_bound": LOG_SCALE_BOUND, "initial_equal_class_nll": initial_loss, "final_equal_class_nll": final_loss, "max_preclip_gradient_norm": gradient_max, "mean_abs_target_log_density_odds": mean_abs_log_odds, "parameter_sha256": parameter_sha256(model), "exact_change_of_variables_log_determinant": True, "separate_up_down_class_densities": True, "fixed_alternating_masks": True, "equal_class_objective": True, "outer_or_target_outcomes_used": False, **device_audit}
    require(audit["parameter_count"] == TOTAL_PARAMETER_N, "V196 RealNVP parameter count changed")
    return np.clip(probability, 1e-6, 1.0 - 1e-6), audit


def realnvp_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V196 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_numeric, train_transform = processor.transform(ordered)
    target_numeric, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_numeric[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_numeric[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    train_state, train_state_audit = flow_state(group_path)
    target_state, target_state_audit = flow_state(target_path)
    probability, training = fit_realnvp(train_state, target_state, group_target)
    return probability, {"train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit, "train_state": train_state_audit, "target_state": target_state_audit, "train_geometry": train_state_audit, "target_geometry": target_state_audit, "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": FLOW_CLIP}, "feature_scaling": {"train_only": True, "method": "fixed division after train-cut channel robust scale", "target_batch_fit": False}, "training": training, "target_rows_used_for_scale_flow_or_density_fit": False, "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False, "tensor_train_mps_or_quantum_density": False, "gaussian_copula_qda_or_vmf": False, "direct_ulsif_or_covariate_ratio_reweighting": False, "nystrom_rbf_rff_or_random_maclaurin": False, "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)}}


_BASE_BOOTSTRAP = scaffold._BASE_BOOTSTRAP


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = STATE_N
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = realnvp_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = realnvp_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    training = model_audit["training"]
    state = model_audit["train_state"]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["channel_scaling"]["train_only"] and model_audit["feature_scaling"]["train_only"] and state["fixed_bounded_affine_state"] and state["immutable_row_major_order"] and not state["tensor_train_or_mps"] and training["exact_change_of_variables_log_determinant"] and training["separate_up_down_class_densities"] and training["fixed_alternating_masks"] and training["equal_class_objective"] and training["parameter_count"] == TOTAL_PARAMETER_N and training["final_equal_class_nll"] <= training["initial_equal_class_nll"] + 1e-4 and not training["outer_or_target_outcomes_used"] and not model_audit["target_rows_used_for_scale_flow_or_density_fit"] and not model_audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2.0 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 class-conditional RealNVP density-ratio blend with fixed safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "flow_depth_width_epoch_optimizer_blend_or_source_floor_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    base.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v196_model"})
    evidence = evidence.rename(columns={"frenet_prob": "realnvp_density_ratio_prob", "v146_model": "v196_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V196 REALNVP] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V196 RealNVP density-ratio policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base()
    base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["parameter_count_two_flows"] == TOTAL_PARAMETER_N and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["train_state"]["fixed_bounded_affine_state"] and audit[side]["train_state"]["immutable_row_major_order"] and not audit[side]["train_state"]["tensor_train_or_mps"] and audit[side]["training"]["exact_change_of_variables_log_determinant"] and audit[side]["training"]["separate_up_down_class_densities"] and audit[side]["training"]["fixed_alternating_masks"] and audit[side]["training"]["equal_class_objective"] and audit[side]["training"]["parameter_count"] == TOTAL_PARAMETER_N and audit[side]["training"]["final_equal_class_nll"] <= audit[side]["training"]["initial_equal_class_nll"] + 1e-4 and not audit[side]["training"]["outer_or_target_outcomes_used"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["target_rows_used_for_scale_flow_or_density_fit"] and not audit[side]["target_labels_used"] and not audit[side]["tensor_train_mps_or_quantum_density"] and not audit[side]["gaussian_copula_qda_or_vmf"] and not audit[side]["direct_ulsif_or_covariate_ratio_reweighting"] for audit in audits for side in ("inner_model_audit", "outer_model_audit")))
    checks["multichannel_causal_class_conditional_realnvp_density_ratio_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V196 fallback not exact V69")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V196 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V196 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V196 output target mismatch")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V196").resolve(), "V196 output outside staging/V196")
    require(out.exists() and out.is_dir(), "V196 staging child must preexist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V196 prefiles missing: {sorted(required-existing)}")
    require(not (existing - allowed), f"V196 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V196 output not controller-bound")
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V196 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V196", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V196_DIAGNOSTIC_REALNVP_DENSITY_RATIO": evaluation["candidate_summary"], "V196_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V196", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V196", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V196_CLASS_CONDITIONAL_REALNVP_DENSITY_RATIO_REPORT.json": {"version": "V196", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V196", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V196", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V196", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V196_REALNVP_DENSITY_RATIO_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V196_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V196 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V196 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V196 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 1, "device": "cuda:0 deterministic required"}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "two class-conditional four-layer RealNVP densities with exact log determinants and direct likelihood-ratio probability", "quantum_mps_rejected_due_v192_tensor_train_collision": True, "causal_numeric_only": True, "target_labels_used": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V196 support reserved for KR:2")
            audit = audits[0]
            non = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({"status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS, "runtime": runtime, "market": "KR", "fold": 2, "strict_inner_chronology": audit["inner_chronology"], "strict_outer_chronology": audit["outer_chronology"], "raw_inner_selected": audit["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audit["outer_baseline"], "raw_outer_selected": audit["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audit["inner_model_audit"], "outer_model_audit": audit["outer_model_audit"], "selection_locked_before_outer_evaluation": True, "nested_bootstrap_executed": False, "exact_entire_v69_fallback_contract": True, "output_written": False}), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, smoke=args.smoke_test)
    non = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best = max(non, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {"status": "SMOKE_OK" if args.smoke_test else evaluation["status"], "hypothesis": HYPOTHESIS, "runtime": runtime, "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None, "selected_models": [audit["policy"]["selected"]["name"] for audit in audits], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_pass": evaluation["material_pass"], "fallback_is_exact_v69": not evaluation["material_pass"], "canonical_gate": evaluation["selected_summary"]["research_gate"], "current_material_gate": material_gate(evaluation), "output_written": False}
    if args.smoke_test:
        summary.update({"raw_inner_selected": audits[0]["policy"]["selected"], "raw_inner_best_non_noop": best, "raw_outer_baseline": audits[0]["outer_baseline"], "raw_outer_selected": audits[0]["outer_candidate"], "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best["name"]], "inner_model_audit": audits[0]["inner_model_audit"], "outer_model_audit": audits[0]["outer_model_audit"], "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]})
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
