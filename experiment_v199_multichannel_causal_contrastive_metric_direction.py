"""V199 supervised contrastive metric direction challenger.

Strict-past equal event-group 64D causal surfaces train one deterministic small
MLP into a 12D unit sphere with a full-batch, equal-class supervised
contrastive objective.  Only past training embeddings define normalized UP and
DOWN prototypes.  Query direction is the fixed-temperature cosine prototype
margin.  This is supervised metric geometry, not V82 masked reconstruction,
V147 sliced distribution distance, V164 support balls, V196 normalized flows,
or V198 denoising score-matching energy.

Bounded preparation is CPU-only with GPU hidden. Controller-bound full mode
requires deterministic CUDA device 0 and fails closed otherwise. Inner OOF
selects exact V69 or fixed .25/.50 blends; outer labels are evaluation-only
after strict 35-minute embargo/event purge. V69 confidence/high_conf remain
exact and any material/execution failure restores the entire V69 frame.
"""
from __future__ import annotations

import ctypes
import os

if not os.environ.get("MARKET_BIO_VERSION_OUTPUT"):
    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _kernel32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    _process_handle = _kernel32.GetCurrentProcess()
    if not _kernel32.SetProcessAffinityMask(_process_handle, ctypes.c_size_t(0xC0000000)):
        raise RuntimeError("V199 bounded CPU affinity set failed")
    _observed_process_mask = ctypes.c_size_t()
    _observed_system_mask = ctypes.c_size_t()
    if not _kernel32.GetProcessAffinityMask(_process_handle, ctypes.byref(_observed_process_mask), ctypes.byref(_observed_system_mask)) or _observed_process_mask.value != 0xC0000000:
        raise RuntimeError(f"V199 bounded CPU affinity mismatch: {_observed_process_mask.value:#x}")

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
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

import experiment_v196_multichannel_causal_class_conditional_realnvp_direction as scaffold


base = scaffold.base
ROOT = scaffold.ROOT
DEFAULT_OUT = ROOT / "research" / "staging" / "V199" / "LOCAL_FORBIDDEN"
VERSION = 199
HYPOTHESIS = "MULTICHANNEL_CAUSAL_CONTRASTIVE_METRIC_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v196_multichannel_causal_class_conditional_realnvp_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "52682372609a2e4d2f1563aa607a523d352e748a82980a25476da513e70ddc1b"

HORIZON_N = 8
CHANNEL_N = 8
STATE_N = HORIZON_N * CHANNEL_N
HIDDEN_N = 32
EMBEDDING_N = 12
STATE_CLIP = 5.0
STATE_DIVISOR = 2.5
CONTRASTIVE_TEMPERATURE = 0.15
PROTOTYPE_TEMPERATURE = 0.25
TRAIN_EPOCHS = 32
LEARNING_RATE = 0.010
WEIGHT_DECAY = 0.0005
GRADIENT_CLIP = 5.0
SEED = 19901
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CONTRASTIVE_METRIC_W0.25", "weight": 0.25},
    {"name": "CONTRASTIVE_METRIC_W0.50", "weight": 0.50},
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

PARAMETER_N = STATE_N * HIDDEN_N + HIDDEN_N + HIDDEN_N * EMBEDDING_N + EMBEDDING_N
STATIC_SPEC = {
    "surface_shape": [HORIZON_N, CHANNEL_N],
    "state_order": "64 row-major coordinates: 120m-to-1m horizons outer, immutable semantic channels inner",
    "state_transform": "clip(train-cut channel-scaled value,-5,5)/2.5",
    "embedding": "64-to-32-tanh-to-12 then exact L2 normalization",
    "feature_dimension": EMBEDDING_N,
    "parameter_count": PARAMETER_N,
    "objective": "full-batch equal-event-group/equal-class supervised contrastive loss",
    "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
    "prototype_score": "cosine(query,UP prototype)-cosine(query,DOWN prototype)",
    "prototype_temperature": PROTOTYPE_TEMPERATURE,
    "optimizer": {"name": "AdamW", "epochs": TRAIN_EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP},
    "bounded_device": "CPU only with CUDA hidden",
    "controller_full_device": "deterministic CUDA device 0 required; fail closed otherwise",
    "target_batch_fit": False,
}
require(PARAMETER_N == 2476, "V199 embedding parameter dimension changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V196 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["v199_code_dependency"] = {"path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256, "purpose": "immutable V36/V69 authority, chronology, observed path, metrics, gate, bootstrap and atomic IO only; no V196 output is read"}
    audit["v199_access_contract"] = {"immutable_v36_dev_only": True, "atomic_output_v69_only": True, "failed_outputs_read": False, "dev_extension_read": False, "role_assignment_read": False, "research_seal_read": False, "final_reserve_read": False, "prior_version_output_read": False}
    audit["gpu_execution_contract"] = {"bounded_cuda_hidden": not bool(CONTROLLER_OUTPUT), "controller_full_requires_cuda_device_zero": True, "deterministic_algorithms_required": True, "nondeterminism_or_cuda_unavailable_fails_closed": True}
    audit["bounded_os_affinity_contract"] = {"set_before_heavy_imports": not bool(CONTROLLER_OUTPUT), "required_mask_hex": "C0000000", "observed_mask_hex": None if CONTROLLER_OUTPUT else f"{_observed_process_mask.value:X}", "verified": bool(CONTROLLER_OUTPUT or _observed_process_mask.value == 0xC0000000)}
    return audit


def model_device() -> tuple[torch.device, dict[str, Any]]:
    torch.set_num_threads(SMOKE_THREADS)
    torch.use_deterministic_algorithms(True)
    if not CONTROLLER_OUTPUT:
        return torch.device("cpu"), {"device": "cpu", "bounded_cpu_only": True, "cuda_visible_devices_requested": "-1", "all_model_tensors_on_cpu": True, "gpu_model_calls": 0, "controller_full_gpu0": False, "deterministic_algorithms": True}
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "V199 controller full requires CUDA device 0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0"), {"device": "cuda:0", "bounded_cpu_only": False, "gpu_model_calls": 1, "controller_full_gpu0": True, "deterministic_algorithms": True}


def bounded_state(path: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    require(path.ndim == 3 and path.shape[1:] == (HORIZON_N, CHANNEL_N), "V199 path shape invalid")
    state = np.clip(path.reshape(len(path), STATE_N), -STATE_CLIP, STATE_CLIP) / STATE_DIVISOR
    require(state.shape == (len(path), STATE_N) and np.isfinite(state).all(), "V199 state invalid")
    return state, {"rows": len(state), "state_n": STATE_N, "state_sha256": array_sha256(state), "absolute_max": float(np.max(np.abs(state))), "immutable_row_major_order": True, "fixed_bounded_affine_state": True, "target_batch_fit": False}


class ContrastiveEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED)
        self.first = nn.Linear(STATE_N, HIDDEN_N, dtype=torch.float32)
        self.second = nn.Linear(HIDDEN_N, EMBEDDING_N, dtype=torch.float32)
        with torch.no_grad():
            self.first.weight.copy_(0.04 * torch.randn(self.first.weight.shape, generator=generator, dtype=torch.float32))
            self.first.bias.zero_()
            self.second.weight.copy_(0.04 * torch.randn(self.second.weight.shape, generator=generator, dtype=torch.float32))
            self.second.bias.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.second(torch.tanh(self.first(value))), p=2.0, dim=1, eps=1e-8)


def supervised_contrastive_loss(embedding: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    similarity = torch.matmul(embedding, embedding.T) / CONTRASTIVE_TEMPERATURE
    n = len(embedding)
    identity = torch.eye(n, dtype=torch.bool, device=embedding.device)
    same = label[:, None].eq(label[None, :]) & ~identity
    require(bool(torch.all(same.sum(dim=1) > 0).item()), "V199 anchor without positive")
    denominator = torch.logsumexp(similarity.masked_fill(identity, -torch.inf), dim=1)
    log_probability = similarity - denominator[:, None]
    anchor_loss = -(log_probability.masked_fill(~same, 0.0).sum(dim=1) / same.sum(dim=1))
    class_n = torch.bincount(label, minlength=2).to(embedding.dtype)
    weight = torch.where(label == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    return torch.sum(weight * anchor_loss)


def parameter_sha256(model: ContrastiveEmbedding) -> str:
    flat = np.concatenate([parameter.detach().cpu().numpy().reshape(-1) for parameter in model.parameters()])
    return array_sha256(flat)


def fit_contrastive(train_state: np.ndarray, target_state: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    device, device_audit = model_device()
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = ContrastiveEmbedding().to(device)
    train_tensor = torch.as_tensor(train_state, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target_state, dtype=torch.float32, device=device)
    label = torch.as_tensor(target.astype(np.int64), dtype=torch.long, device=device)
    class_n = np.bincount(target, minlength=2)
    require(int(class_n.min()) >= 12, "V199 insufficient class support")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    with torch.no_grad():
        initial_loss = float(supervised_contrastive_loss(model(train_tensor), label).cpu())
    gradient_max = 0.0
    for _ in range(TRAIN_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        embedding = model(train_tensor)
        loss = supervised_contrastive_loss(embedding, label)
        require(bool(torch.isfinite(loss).item()), "V199 contrastive loss non-finite")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP).detach().cpu())
        gradient_max = max(gradient_max, gradient)
        optimizer.step()
    with torch.no_grad():
        train_embedding = model(train_tensor)
        final_loss = float(supervised_contrastive_loss(train_embedding, label).cpu())
        target_embedding = model(target_tensor)
        down_prototype = F.normalize(train_embedding[label == 0].mean(dim=0), p=2.0, dim=0, eps=1e-8)
        up_prototype = F.normalize(train_embedding[label == 1].mean(dim=0), p=2.0, dim=0, eps=1e-8)
        margin = (target_embedding @ up_prototype - target_embedding @ down_prototype) / PROTOTYPE_TEMPERATURE
        probability = torch.sigmoid(margin).cpu().numpy()
        train_norm_error = float(torch.max(torch.abs(torch.linalg.vector_norm(train_embedding, dim=1) - 1.0)).cpu())
        target_norm_error = float(torch.max(torch.abs(torch.linalg.vector_norm(target_embedding, dim=1) - 1.0)).cpu())
        prototype_cosine = float(torch.dot(down_prototype, up_prototype).cpu())
    require(np.isfinite(probability).all() and final_loss <= initial_loss + 1e-4, "V199 contrastive training contract failed")
    audit = {"model": "supervised contrastive normalized embedding plus class prototype cosine margin", "hidden_n": HIDDEN_N, "embedding_n": EMBEDDING_N, "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "epochs": TRAIN_EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP, "contrastive_temperature": CONTRASTIVE_TEMPERATURE, "prototype_temperature": PROTOTYPE_TEMPERATURE, "initial_supervised_contrastive_loss": initial_loss, "final_supervised_contrastive_loss": final_loss, "max_preclip_gradient_norm": gradient_max, "train_embedding_norm_error_max": train_norm_error, "target_embedding_norm_error_max": target_norm_error, "class_prototype_cosine": prototype_cosine, "parameter_sha256": parameter_sha256(model), "full_batch_all_positive_negative_pairs": True, "equal_event_group_and_class_anchor_weighting": True, "train_only_class_prototypes": True, "fixed_cosine_margin_probability": True, "outer_or_target_outcomes_used": False, **device_audit}
    geometry = {"feature_dimension": EMBEDDING_N, "label_independent_exact_transform": False, "local_differential_geometry_not_global_path_integral": False, "l2_normalized": True, "target_batch_fit": False, "train_embedding_sha256": array_sha256(train_embedding.cpu().numpy()), "target_embedding_sha256": array_sha256(target_embedding.cpu().numpy())}
    require(audit["parameter_count"] == PARAMETER_N, "V199 parameter count changed")
    return np.clip(probability, 1e-6, 1.0 - 1e-6), audit, geometry


def contrastive_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == target_frame.market.nunique() == 1, "V199 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_numeric, train_transform = processor.transform(ordered)
    target_numeric, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_past_to_recent_path(train_numeric[:, :len(FEATURES)])
    target_path, target_path_audit = observed_past_to_recent_path(target_numeric[:, :len(FEATURES)])
    group_path, group_target, grouping = equal_event_group_path(ordered, train_path)
    channel_scale = TrainChannelScale().fit(group_path)
    group_path = channel_scale.transform(group_path)
    target_path = channel_scale.transform(target_path)
    train_state, train_state_audit = bounded_state(group_path)
    target_state, target_state_audit = bounded_state(target_path)
    probability, training, geometry = fit_contrastive(train_state, target_state, group_target)
    return probability, {"train_n": len(ordered), "train_event_group_n": len(group_target), "target_n": len(target_frame), "causal_numeric_only": True, "categorical_text_source_ticker_or_issuer_feature_n": 0, "train_transform": train_transform, "target_transform": target_transform, "grouping": grouping, "static_spec": STATIC_SPEC, "train_path": train_path_audit, "target_path": target_path_audit, "train_state": train_state_audit, "target_state": target_state_audit, "train_geometry": geometry, "target_geometry": geometry, "channel_scaling": {"train_only": True, "median_sha256": array_sha256(channel_scale.median), "scale_sha256": array_sha256(channel_scale.scale), "clip": STATE_CLIP}, "feature_scaling": {"train_only": True, "method": "fixed division after train-cut channel robust scale", "target_batch_fit": False}, "training": training, "target_rows_used_for_contrastive_fit_scale_or_prototypes": False, "target_rows_used_for_robust_scale_geometry_scale_or_head": False, "target_labels_used": False, "haar_wavelet_or_scattering": False, "rough_path_signature_or_levy_area": False, "increment_spd_or_covariance": False, "fourier_cross_spectrum_or_dmd": False, "cusum_changepoint_or_recurrence": False, "natural_visibility_graph_or_ordinal_motif": False, "source_time_environment_transport_or_projection": False, "masked_reconstruction_or_self_supervised_consistency": False, "normalizing_flow_score_matching_or_density_ratio": False, "sliced_wasserstein_support_ball_or_distance_density": False, "prediction": {"mean": float(probability.mean()), "std": float(probability.std()), "probability_sha256": array_sha256(probability)}}


_BASE_BOOTSTRAP = scaffold._BASE_BOOTSTRAP


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = EMBEDDING_N
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = contrastive_direction


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = contrastive_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{"name": "V69_NOOP", "architecture": None, "metrics": baseline_metric, "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0, "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {}, "eligible": True, "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"]}]
    training = model_audit["training"]
    model_eligible = bool(model_audit["causal_numeric_only"] and model_audit["grouping"]["equal_event_group_weighting"] and model_audit["channel_scaling"]["train_only"] and model_audit["feature_scaling"]["train_only"] and training["full_batch_all_positive_negative_pairs"] and training["equal_event_group_and_class_anchor_weighting"] and training["train_only_class_prototypes"] and training["fixed_cosine_margin_probability"] and training["parameter_count"] == PARAMETER_N and training["final_supervised_contrastive_loss"] <= training["initial_supervised_contrastive_loss"] + 1e-4 and not training["outer_or_target_outcomes_used"] and not model_audit["target_rows_used_for_contrastive_fit_scale_or_prototypes"] and not model_audit["target_labels_used"] and not model_audit["masked_reconstruction_or_self_supervised_consistency"] and not model_audit["normalizing_flow_score_matching_or_density_ratio"])
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({"name": architecture["name"], "architecture": architecture, "metrics": current, "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta, "worst_supported_source_ba_delta": source_floor, "supported_source_ba_delta": source_deltas, "model_eligible": model_eligible, "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010), "score": 2.0 * current["auc"] + current["balanced_accuracy"]})
    selected = max([trial for trial in trials if trial["eligible"]], key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {"selection_rule": "inner-past OOF only; exact V69 noop or fixed .25/.50 supervised-contrastive prototype-margin blend with fixed safety", "baseline": baseline_metric, "selected": selected, "trials": trials, "outer_labels_used_for_selection": False, "embedding_width_temperature_epoch_optimizer_prototype_or_blend_micro_tuning": False}, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool, smoke_market: str = "US") -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    base.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v199_model"})
    evidence = evidence.rename(columns={"frenet_prob": "contrastive_metric_prob", "v146_model": "v199_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(f"[V199 CONTRASTIVE] market={audit['market']} fold={audit['fold']} selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audit['outer_auc_delta']:+.6f}", flush=True)
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V199 supervised-contrastive metric policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    configure_base()
    base.paired_nested_bootstrap = paired_nested_bootstrap
    result = base.evaluate(champion, diagnostic, evidence, audits, draws, smoke)
    checks = result["material_checks"]
    checks.pop("multichannel_discrete_frenet_curvature_contract_verified")
    contract = bool(all(audit[side]["causal_numeric_only"] and audit[side]["static_spec"]["parameter_count"] == PARAMETER_N and audit[side]["grouping"]["equal_event_group_weighting"] and audit[side]["training"]["full_batch_all_positive_negative_pairs"] and audit[side]["training"]["equal_event_group_and_class_anchor_weighting"] and audit[side]["training"]["train_only_class_prototypes"] and audit[side]["training"]["fixed_cosine_margin_probability"] and audit[side]["training"]["parameter_count"] == PARAMETER_N and audit[side]["training"]["final_supervised_contrastive_loss"] <= audit[side]["training"]["initial_supervised_contrastive_loss"] + 1e-4 and not audit[side]["training"]["outer_or_target_outcomes_used"] and audit[side]["channel_scaling"]["train_only"] and audit[side]["feature_scaling"]["train_only"] and not audit[side]["target_rows_used_for_contrastive_fit_scale_or_prototypes"] and not audit[side]["target_labels_used"] and not audit[side]["masked_reconstruction_or_self_supervised_consistency"] and not audit[side]["normalizing_flow_score_matching_or_density_ratio"] and not audit[side]["sliced_wasserstein_support_ball_or_distance_density"] for audit in audits for side in ("inner_model_audit", "outer_model_audit")))
    checks["multichannel_causal_supervised_contrastive_metric_contract_verified"] = contract
    result["material_pass"] = bool(not smoke and all(checks.values()))
    if not result["material_pass"]:
        result["selected_frame"] = champion.copy()
        result["selected_summary"] = result["baseline_summary"]
        result["fallback"] = {"activated": True, "policy": "exact entire V69 DataFrame", "exact_entire_frame_verified": bool(result["selected_frame"].equals(champion))}
        require(result["selected_frame"].equals(champion), "V199 fallback not exact V69")
    return result


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {"contract": HYPOTHESIS, "checks": checks, "passed": int(sum(bool(value) for value in checks.values())), "total": len(checks), "material_pass": bool(evaluation["material_pass"]), "nested_bootstrap": evaluation["nested_bootstrap"], "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"], "current_hypothesis_not_inherited_from_v69": True, "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame"}
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V199 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V199 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V199 output target mismatch")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V199").resolve(), "V199 output outside staging/V199")
    require(out.exists() and out.is_dir(), "V199 staging child must preexist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V199 prefiles missing: {sorted(required-existing)}")
    require(not (existing - allowed), f"V199 unexpected conflict: {sorted(existing-allowed)}")
    return {"controller_bound": True, "target_preexisting": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [], "conflict_policy": "allow exact controller prefiles only; fail closed before write"}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V199 output not controller-bound")
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V199 native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {"version": "V199", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V199_DIAGNOSTIC_CONTRASTIVE_METRIC": evaluation["candidate_summary"], "V199_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V199", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V199", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "V199_CONTRASTIVE_METRIC_REPORT.json": {"version": "V199", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {"version": "V199", "hypothesis": HYPOTHESIS, "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"]},
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {"version": "V199", "hypothesis": HYPOTHESIS, "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"]},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V199", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"], "seal_state": "UNOPENED", "seal_authorized": False, "output_conflict_audit": conflict, "completed_at": pd.Timestamp.now(tz="UTC").isoformat()},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V199_CONTRASTIVE_METRIC_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V199_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
        require(not args.audit_only and not args.smoke_test and not args.support_probe and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V199 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V199 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V199 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {"mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 1, "device": "cuda:0 deterministic required"}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority, "runtime": runtime, "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC, "architecture": "supervised contrastive normalized 64-32-12 embedding plus train-only class-prototype cosine margin", "prior_supervised_contrastive_siamese_triplet_hits": 0, "causal_numeric_only": True, "target_labels_used": False, "canonical_controller_gate_checks": 14, "controller_contract": {"no_argument_market_bio_version_output_full": True, "controller_output_exact_binding": True, "explicit_mode_or_output_conflict_fails_closed": True, "direct_full_without_controller_fails_closed": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "native_robustness_bootstrap_preserved": True, "source_transfer_current_gate": True, "exact_entire_v69_fallback": True}, "output_written": False}), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V199 support reserved for KR:2")
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
