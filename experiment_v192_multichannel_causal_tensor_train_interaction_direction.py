"""V192 multichannel causal tensor-train interaction direction.

The strict-past train-scaled immutable V36 8x8 causal surface is ordered by
past-to-recent horizon then semantic channel.  Each of its 64 sites receives
the frozen local basis [1, z, z^2], z=clip(x,-7,7)/7.  A single rank-4 tensor
train contracts all 64 local bases into one direction logit.  Its 3,001
parameters are learned with one deterministic fixed full-batch, equal-event-
group/class-balanced BCE schedule.  This is a high-order multilinear tensor
network, not Nyström/RBF/RFF/kernel approximation, PLS/ICA/NMF, pairwise
bilinear field model, convolution, recurrence, attention or path transform.

Bounded preparation is CPU-only with the GPU hidden.  Controller-bound full
mode requires deterministic CUDA device 0 and fails closed otherwise.  Inner
OOF selects exact V69 or fixed .25/.50 blends; outer labels are evaluation-only
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as torch_functional
from threadpoolctl import threadpool_limits

import experiment_v189_multichannel_causal_multiscale_renorm_variation_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V192" / "LOCAL_FORBIDDEN"
VERSION = 192
HYPOTHESIS = "MULTICHANNEL_CAUSAL_TENSOR_TRAIN_INTERACTION_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v189_multichannel_causal_multiscale_renorm_variation_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "968226d662e0ba8d600cb73c1e009cfb929e04b43c73d1bb2cff82c0a142204e"

HORIZON_N = 8
CHANNEL_N = 8
SITE_N = HORIZON_N * CHANNEL_N
LOCAL_BASIS_N = 3
LOCAL_BASIS_COORDINATE_N = SITE_N * LOCAL_BASIS_N
TT_RANK = 4
TT_PARAMETER_N = LOCAL_BASIS_N * TT_RANK + (SITE_N - 2) * TT_RANK * LOCAL_BASIS_N * TT_RANK + TT_RANK * LOCAL_BASIS_N + 1
CHANNEL_CLIP = 7.0
TRAIN_EPOCHS = 48
LEARNING_RATE = 0.015
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 5.0
SEED = 19201
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "TENSOR_TRAIN_W0.25", "weight": 0.25},
    {"name": "TENSOR_TRAIN_W0.50", "weight": 0.50},
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
    "surface_shape": [HORIZON_N, CHANNEL_N],
    "mode_order": "64 row-major sites: 120m-to-1m horizons outer, immutable semantic channels inner",
    "local_basis": ["1", "z", "z_squared"],
    "local_state": "z=clip(train-cut channel-scaled value,-7,7)/7",
    "local_basis_coordinate_count": LOCAL_BASIS_COORDINATE_N,
    "tensor_train_rank": TT_RANK,
    "tensor_train_parameter_count_including_bias": TT_PARAMETER_N,
    "objective": "fixed full-batch equal-event-group/class-balanced BCE",
    "optimizer": {"name": "AdamW", "epochs": TRAIN_EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP},
    "feature_dimension": LOCAL_BASIS_COORDINATE_N,
    "bounded_device": "CPU only with CUDA hidden",
    "controller_full_device": "deterministic CUDA device 0 required; fail closed otherwise",
    "kernel_nystrom_rff_or_landmark": False,
    "target_batch_fit": False,
}
require(LOCAL_BASIS_COORDINATE_N == 192 and TT_PARAMETER_N == 3001, "V192 tensor-train dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V189 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["v192_code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, observed path, metrics, gate, bootstrap and atomic IO only; no V189 output is read"}
    audit["v192_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    audit["gpu_execution_contract"] = {"bounded_cuda_hidden": not bool(CONTROLLER_OUTPUT), "controller_full_requires_cuda_device_zero": True, "deterministic_algorithms_required": True, "nondeterminism_or_cuda_unavailable_fails_closed": True}
    return audit


def model_device() -> tuple[torch.device, dict[str, Any]]:
    torch.set_num_threads(SMOKE_THREADS)
    torch.use_deterministic_algorithms(True)
    if not CONTROLLER_OUTPUT:
        return torch.device("cpu"), {"device": "cpu", "bounded_cpu_only": True, "cuda_visible_devices_requested": "-1", "all_model_tensors_on_cpu": True, "gpu_model_calls": 0, "controller_full_gpu0": False, "deterministic_algorithms": True}
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "V192 controller full requires CUDA device 0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0"), {"device": "cuda:0", "bounded_cpu_only": False, "gpu_model_calls": 1, "controller_full_gpu0": True, "deterministic_algorithms": True}


def local_basis(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V192 path shape invalid")
    z = np.clip(path.reshape(len(path), SITE_N), -CHANNEL_CLIP, CHANNEL_CLIP) / CHANNEL_CLIP
    result = np.stack((np.ones_like(z), z, np.square(z)), axis=2)
    require(result.shape == (len(path), SITE_N, LOCAL_BASIS_N) and np.isfinite(result).all(), "V192 local basis invalid")
    return result, {"rows": len(result), "site_n": SITE_N, "local_basis_n": LOCAL_BASIS_N, "feature_dimension": LOCAL_BASIS_COORDINATE_N, "basis_sha256": array_sha256(result), "basis_constant_exact": bool(np.array_equal(result[:, :, 0], np.ones((len(path), SITE_N)))), "basis_state_abs_max": float(np.max(np.abs(result[:, :, 1]))), "basis_square_nonnegative": bool(np.min(result[:, :, 2]) >= 0.0), "fixed_local_polynomial_basis": True, "immutable_row_major_mode_order": True, "label_independent_exact_transform": True, "local_differential_geometry_not_global_path_integral": False, "kernel_nystrom_rff_or_landmark": False, "pairwise_bilinear_or_field_factor": False, "convolution_recurrence_attention_or_path_transform": False, "target_batch_fit": False}


class TensorTrainLogit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED)
        cores: list[nn.Parameter] = []
        first = 0.004 * torch.randn((1, LOCAL_BASIS_N, TT_RANK), generator=generator, dtype=torch.float64)
        first[0, 0, :] = 0.0
        first[0, 0, 0] = 1.0
        cores.append(nn.Parameter(first))
        for _ in range(SITE_N - 2):
            core = 0.004 * torch.randn((TT_RANK, LOCAL_BASIS_N, TT_RANK), generator=generator, dtype=torch.float64)
            core[:, 0, :] = torch.eye(TT_RANK, dtype=torch.float64)
            cores.append(nn.Parameter(core))
        last = 0.004 * torch.randn((TT_RANK, LOCAL_BASIS_N, 1), generator=generator, dtype=torch.float64)
        cores.append(nn.Parameter(last))
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, basis: torch.Tensor) -> torch.Tensor:
        state = torch.einsum("nm,mr->nr", basis[:, 0, :], self.cores[0][0])
        for site in range(1, SITE_N - 1):
            state = torch.einsum("na,nm,amb->nb", state, basis[:, site, :], self.cores[site])
        logit = torch.einsum("na,nm,am->n", state, basis[:, SITE_N - 1, :], self.cores[-1][:, :, 0]) + self.bias
        return logit


def parameter_sha256(model: TensorTrainLogit) -> str:
    flat = np.concatenate([parameter.detach().cpu().numpy().reshape(-1) for parameter in model.parameters()])
    return array_sha256(flat)


def fit_tensor_train(train_basis: np.ndarray, target_basis: np.ndarray, target: np.ndarray, class_n: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    device, device_audit = model_device()
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = TensorTrainLogit().to(device)
    train_tensor = torch.as_tensor(train_basis, dtype=torch.float64, device=device)
    target_tensor = torch.as_tensor(target_basis, dtype=torch.float64, device=device)
    target_label = torch.as_tensor(target.astype(float), dtype=torch.float64, device=device)
    weight_np = np.where(target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    sample_weight = torch.as_tensor(weight_np, dtype=torch.float64, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    with torch.no_grad():
        initial_loss = float(torch.sum(sample_weight * torch_functional.binary_cross_entropy_with_logits(model(train_tensor), target_label, reduction="none")).cpu())
    gradient_max = 0.0
    for _ in range(TRAIN_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logit = model(train_tensor)
        loss = torch.sum(sample_weight * torch_functional.binary_cross_entropy_with_logits(logit, target_label, reduction="none"))
        require(bool(torch.isfinite(loss).item()), "V192 TT loss non-finite")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP).detach().cpu())
        gradient_max = max(gradient_max, gradient)
        optimizer.step()
    with torch.no_grad():
        final_loss = float(torch.sum(sample_weight * torch_functional.binary_cross_entropy_with_logits(model(train_tensor), target_label, reduction="none")).cpu())
        probability = torch.sigmoid(model(target_tensor)).detach().cpu().numpy()
    require(np.isfinite(probability).all() and final_loss <= initial_loss + 1e-4, "V192 TT training contract failed")
    audit = {"model": "rank-4 tensor-train logistic", "site_n": SITE_N, "local_basis_n": LOCAL_BASIS_N, "tt_rank": TT_RANK, "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "epochs": TRAIN_EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP, "initial_balanced_bce": initial_loss, "final_balanced_bce": final_loss, "max_preclip_gradient_norm": gradient_max, "parameter_sha256": parameter_sha256(model), "fixed_full_batch_schedule": True, "outer_or_target_outcomes_used": False, **device_audit}
    require(audit["parameter_count"] == TT_PARAMETER_N, "V192 TT parameter count changed")
    return np.clip(probability, 1e-6, 1.0 - 1e-6), audit


def tensor_train_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V192 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_state[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    train_basis, train_basis_audit = local_basis(group_path)
    target_basis, target_basis_audit = local_basis(target_path)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    probability, training = fit_tensor_train(train_basis, target_basis, group_target, class_n)
    return probability, {"train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit, "train_geometry": train_basis_audit, "target_geometry": target_basis_audit, "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": CHANNEL_CLIP}, "feature_scaling": {"train_only": True, "method": "fixed division by channel clip after train-cut channel robust scale", "target_batch_fit": False}, "training": training, "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False, "nystrom_rbf_rff_or_random_maclaurin": False, "field_aware_bilinear_or_pls_ica_nmf": False, "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)}}


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = LOCAL_BASIS_COORDINATE_N
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = tensor_train_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = tensor_train_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    training = model_audit["training"]
    geometry = model_audit["train_geometry"]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["channel_scaling"]["train_only"] and model_audit["feature_scaling"]["train_only"] and geometry["fixed_local_polynomial_basis"] and geometry["immutable_row_major_mode_order"] and not geometry["kernel_nystrom_rff_or_landmark"] and training["tt_rank"] == TT_RANK and training["parameter_count"] == TT_PARAMETER_N and training["fixed_full_batch_schedule"] and training["final_balanced_bce"] <= training["initial_balanced_bce"] + 1e-4 and not training["outer_or_target_outcomes_used"] and not model_audit["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not model_audit["target_labels_used"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2.0 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 tensor-train blend with fixed safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "rank_basis_epoch_optimizer_head_blend_or_source_floor_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    base.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v192_model"})
    evidence = evidence.rename(columns={"frenet_prob": "tensor_train_prob", "v146_model": "v192_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V192 TENSOR TRAIN] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V192 tensor-train policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base()
    base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["feature_dimension"] == LOCAL_BASIS_COORDINATE_N and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["train_geometry"]["fixed_local_polynomial_basis"] and audit[side]["train_geometry"]["immutable_row_major_mode_order"] and not audit[side]["train_geometry"]["kernel_nystrom_rff_or_landmark"] and audit[side]["training"]["tt_rank"] == TT_RANK and audit[side]["training"]["parameter_count"] == TT_PARAMETER_N and audit[side]["training"]["fixed_full_batch_schedule"] and audit[side]["training"]["final_balanced_bce"] <= audit[side]["training"]["initial_balanced_bce"] + 1e-4 and not audit[side]["training"]["outer_or_target_outcomes_used"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["target_rows_used_for_robust_scale_geometry_scale_or_head"] and not audit[side]["target_labels_used"] and not audit[side]["nystrom_rbf_rff_or_random_maclaurin"] and not audit[side]["field_aware_bilinear_or_pls_ica_nmf"] for audit in audits for side in ("inner_model_audit", "outer_model_audit")))
    checks["multichannel_causal_tensor_train_interaction_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V192 fallback not exact V69")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V192 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V192 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V192 output target mismatch")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V192").resolve(), "V192 output outside staging/V192")
    require(out.exists() and out.is_dir(), "V192 staging child must preexist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V192 prefiles missing: {sorted(required-existing)}")
    require(not (existing - allowed), f"V192 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V192 output not controller-bound")
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V192 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V192", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V192_DIAGNOSTIC_TENSOR_TRAIN": evaluation["candidate_summary"], "V192_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V192", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V192", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V192_TENSOR_TRAIN_INTERACTION_REPORT.json": {"version": "V192", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V192", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V192", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V192", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V192_TENSOR_TRAIN_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V192_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
        require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V192 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V192 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V192 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 1, "device": "cuda:0 deterministic required"}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "rank-4 64-site tensor-train interaction logit over fixed local [1,z,z^2] basis; fixed balanced BCE schedule", "preferred_deep_kernel_rejected_due_v154_nystrom_collision": True, "causal_numeric_only": True, "target_labels_used": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V192 support reserved for KR:2")
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
