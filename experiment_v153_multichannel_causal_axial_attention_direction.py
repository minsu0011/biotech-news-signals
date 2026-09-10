"""V153 multichannel causal axial-attention direction challenger.

For every embargoed same-market train cut, the immutable 37 causal numeric
fields are robust-scaled on past rows only and mapped into the established
eight semantic channels at eight observed pre-event horizons ordered from
120 minutes to 1 minute before the event.  Each scalar cell is embedded from
scratch, receives fixed sinusoidal horizon and channel positions, and passes
through two small blocks that apply horizon-axis self-attention independently
within every channel and then channel-axis self-attention independently within
every horizon.  Mean-plus-recent pooling feeds one direction head trained with
fixed duplicate/class-balanced BCE, seed and schedule.

This is not V57's pretrained US_SEC document-token transformer: V153 uses no
text, tokenizer, pretrained model or source routing, and its attention is
factorized over a causal numeric path for both markets.  It is also distinct
from V150's attention-free token/channel MLP-Mixer.  Inner-past evidence selects
exact V69 or fixed .25/.50 blends; outer labels are evaluation-only after a
strict 35-minute embargo and event-group purge.  V69 confidence/high-confidence
remain exact, and every current material-gate failure restores the exact entire
atomic V69 DataFrame.

Audit, US:2 smoke and KR:2 support use CPU30-31/two threads with zero GPU model
calls.  Full execution is accepted only through no-argument
MARKET_BIO_VERSION_OUTPUT, verifies runtime_limits GPU0 authority and fails
closed before output creation when CUDA GPU0 is unavailable.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _thread_variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_thread_variable] = "2"

import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from threadpoolctl import threadpool_limits

import experiment_v150_multichannel_causal_temporal_channel_mlp_mixer as base


ROOT = base.ROOT
V69_DIR = base.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V153_CAUSAL_AXIAL_ATTENTION_V1"
VERSION = 153
HYPOTHESIS = "MULTICHANNEL_CAUSAL_AXIAL_ATTENTION_DIRECTION_V1"
FEATURES = base.FEATURES
EXPECTED_MARKET_FOLD_ROWS = base.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v150_multichannel_causal_temporal_channel_mlp_mixer.py"
EXPECTED_SCAFFOLD_SHA256 = "b4b0fc2789f27e5556ba1ec18936bafdff93779fea1f31aeee976852a2f30393"

CHANNEL_N = 8
HORIZON_N = 8
MODEL_DIM = 16
HEAD_N = 2
BLOCK_N = 2
FFN_DIM = 32
POOL_N = 2 * MODEL_DIM
SEED = 15301
FULL_EPOCHS = 36
BOUNDED_EPOCHS = 4
LEARNING_RATE = 8.0e-4
WEIGHT_DECAY = 1.0e-3
GRADIENT_CLIP = 2.0
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "CAUSAL_AXIAL_ATTENTION_W0.25", "weight": 0.25},
    {"name": "CAUSAL_AXIAL_ATTENTION_W0.50", "weight": 0.50},
)

require = base.require
sha256 = base.sha256
array_sha256 = base.array_sha256
clean = base.clean
bool_series = base.bool_series
atomic_json = base.atomic_json
atomic_csv = base.atomic_csv
load_authorized = base.load_authorized
blend_probability = base.blend_probability
metric = base.metric
controller = base.controller
v44 = base.v44
numeric = base.numeric

ACTIVE_DEVICE = torch.device("cpu")
ACTIVE_EPOCHS = BOUNDED_EPOCHS
ACTIVE_RUNTIME: dict[str, Any] = {"mode": "unconfigured", "device": "cpu", "gpu_model_calls": 0}


def sinusoidal_positions(length: int, dimension: int, offset: float) -> np.ndarray:
    positions = np.arange(length, dtype=float)[:, None] + offset
    frequency = np.exp(np.arange(0, dimension, 2, dtype=float) * (-np.log(10000.0) / dimension))
    result = np.zeros((length, dimension), dtype=np.float32)
    result[:, 0::2] = np.sin(positions * frequency)
    result[:, 1::2] = np.cos(positions * frequency)
    return result


HORIZON_POSITION = sinusoidal_positions(HORIZON_N, MODEL_DIM, 0.0)
CHANNEL_POSITION = sinusoidal_positions(CHANNEL_N, MODEL_DIM, 17.0)
STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(base.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "all_tokens_strictly_pre_event": True,
    "artificial_origin_or_time_coordinate": False,
    "scalar_cell_embedding": [1, MODEL_DIM],
    "fixed_horizon_sinusoidal_sha256": array_sha256(HORIZON_POSITION),
    "fixed_channel_sinusoidal_sha256": array_sha256(CHANNEL_POSITION),
    "block_n": BLOCK_N,
    "heads": HEAD_N,
    "model_dim": MODEL_DIM,
    "ffn_dim": FFN_DIM,
    "attention_dropout": 0.0,
    "horizon_axis_attention": "independent length-8 attention within each semantic channel",
    "channel_axis_attention": "independent length-8 attention within each observed horizon",
    "pool": "concatenate all-cell mean and nearest-pre-event channel mean",
    "loss": "duplicate/class-balanced weighted BCEWithLogits",
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip": GRADIENT_CLIP,
    "full_epochs": FULL_EPOCHS,
    "bounded_support_epochs": BOUNDED_EPOCHS,
    "seed": SEED,
    "text_or_pretrained_transformer": False,
    "convolution": False,
    "recurrence": False,
    "architecture_or_schedule_grid": False,
}


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V150 utility scaffold changed")
    audit = base.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, causal path interpolation, metrics, canonical gate, blending and atomic IO only; no V150 output is read",
    }
    audit["v153_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
        "prior_version_output_read": False,
    }
    return audit


class AxialAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.horizon_norm = nn.LayerNorm(MODEL_DIM)
        self.horizon_attention = nn.MultiheadAttention(
            MODEL_DIM, HEAD_N, dropout=0.0, batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(MODEL_DIM)
        self.channel_attention = nn.MultiheadAttention(
            MODEL_DIM, HEAD_N, dropout=0.0, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(MODEL_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(MODEL_DIM, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, MODEL_DIM),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.shape[0]
        horizon = self.horizon_norm(tensor).permute(0, 2, 1, 3).reshape(
            batch * CHANNEL_N, HORIZON_N, MODEL_DIM,
        )
        horizon, _ = self.horizon_attention(horizon, horizon, horizon, need_weights=False)
        tensor = tensor + horizon.reshape(batch, CHANNEL_N, HORIZON_N, MODEL_DIM).permute(0, 2, 1, 3)
        channel = self.channel_norm(tensor).reshape(batch * HORIZON_N, CHANNEL_N, MODEL_DIM)
        channel, _ = self.channel_attention(channel, channel, channel, need_weights=False)
        tensor = tensor + channel.reshape(batch, HORIZON_N, CHANNEL_N, MODEL_DIM)
        tensor = tensor + self.ffn(self.ffn_norm(tensor))
        return tensor


class CausalAxialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_embedding = nn.Linear(1, MODEL_DIM)
        self.register_buffer("horizon_position", torch.from_numpy(HORIZON_POSITION)[None, :, None, :])
        self.register_buffer("channel_position", torch.from_numpy(CHANNEL_POSITION)[None, None, :, :])
        self.blocks = nn.ModuleList([AxialAttentionBlock() for _ in range(BLOCK_N)])
        self.final_norm = nn.LayerNorm(MODEL_DIM)
        self.head = nn.Linear(POOL_N, 1)

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        tensor = self.scalar_embedding(path.unsqueeze(-1))
        tensor = tensor + self.horizon_position + self.channel_position
        for block in self.blocks:
            tensor = block(tensor)
        tensor = self.final_norm(tensor)
        pooled = torch.cat([
            tensor.mean(dim=(1, 2)),
            tensor[:, -1, :, :].mean(dim=1),
        ], dim=1)
        return self.head(pooled).squeeze(1)


def configure_execution(bounded: bool) -> dict[str, Any]:
    global ACTIVE_DEVICE, ACTIVE_EPOCHS, ACTIVE_RUNTIME
    if bounded:
        runtime = numeric.configure_bounded_runtime()
        torch.set_num_threads(SMOKE_THREADS)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        ACTIVE_DEVICE = torch.device("cpu")
        ACTIVE_EPOCHS = BOUNDED_EPOCHS
        ACTIVE_RUNTIME = {
            **runtime,
            "mode": "bounded_cpu_support",
            "device": "cpu",
            "epochs": BOUNDED_EPOCHS,
            "gpu_model_calls": 0,
            "torch_cuda_namespace_called": False,
        }
        return ACTIVE_RUNTIME
    runtime_limits = importlib.import_module("runtime_limits")
    runtime_limits.configure()
    status = runtime_limits.status()
    require(runtime_limits.GPU_ENABLED is True, "V153 runtime_limits GPU authority disabled")
    require(status["cuda_visible_devices"] == "0", "V153 runtime_limits did not bind GPU0")
    require(torch.cuda.is_available(), "V153 CUDA GPU0 unavailable under runtime_limits authority")
    require(torch.cuda.device_count() >= 1, "V153 CUDA GPU0 missing")
    ACTIVE_DEVICE = torch.device("cuda:0")
    ACTIVE_EPOCHS = FULL_EPOCHS
    ACTIVE_RUNTIME = {
        **status,
        "mode": "controller_bound_full_gpu0",
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "epochs": FULL_EPOCHS,
        "gpu_model_calls": "authorized full only",
        "runtime_limits_authority_verified": True,
    }
    return ACTIVE_RUNTIME


def set_model_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if ACTIVE_DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def axial_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V153 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = base.observed_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = base.observed_path(target_state[:, :len(FEATURES)])
    target = ordered.y.to_numpy(np.float32)
    sample_weight = numeric.duplicate_class_weights(ordered).astype(np.float32)
    sample_weight = sample_weight / float(np.mean(sample_weight))
    require(np.unique(target).size == 2 and np.isfinite(sample_weight).all(), "V153 weighted target invalid")

    set_model_seed(SEED)
    model = CausalAxialAttention().to(ACTIVE_DEVICE)
    parameter_n = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    train_tensor = torch.from_numpy(train_path).to(ACTIVE_DEVICE)
    target_tensor = torch.from_numpy(target).to(ACTIVE_DEVICE)
    weight_tensor = torch.from_numpy(sample_weight).to(ACTIVE_DEVICE)
    loss_history: list[float] = []
    model.train()
    for _ in range(ACTIVE_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logit = model(train_tensor)
        loss = torch.sum(loss_fn(logit, target_tensor) * weight_tensor) / torch.sum(weight_tensor)
        require(bool(torch.isfinite(loss).item()), "V153 non-finite training loss")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        require(bool(torch.isfinite(gradient_norm).item()), "V153 non-finite gradient")
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.from_numpy(target_path).to(ACTIVE_DEVICE))).cpu().numpy()
    probability = np.clip(probability.astype(float), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V153 probability invalid")
    state_sha = array_sha256(np.concatenate([
        parameter.detach().cpu().numpy().ravel() for parameter in model.parameters()
    ]))
    del train_tensor, target_tensor, weight_tensor, model
    return probability, {
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "train_path": train_path_audit,
        "target_path": target_path_audit,
        "axial_attention": {
            "static_spec": STATIC_SPEC,
            "parameter_n": parameter_n,
            "epochs": ACTIVE_EPOCHS,
            "device": str(ACTIVE_DEVICE),
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "minimum_loss": min(loss_history),
            "model_state_sha256": state_sha,
            "fixed_architecture_seed_and_schedule": True,
            "duplicate_class_balanced_bce": True,
            "factorized_horizon_then_channel_attention": True,
            "fixed_sinusoidal_positions": True,
            "text_or_pretrained_transformer": False,
        },
        "target_rows_used_for_robust_scale_or_training": False,
        "target_labels_used": False,
        "v57_pretrained_sec_document_transformer": False,
        "v150_temporal_channel_mlp_mixer": False,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = axial_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP",
        "architecture": None,
        "metrics": baseline_metric,
        "auc_delta": 0.0,
        "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0,
        "supported_source_ba_delta": {},
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = base.supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({
            "name": architecture["name"],
            "architecture": architecture,
            "metrics": current,
            "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "worst_supported_source_ba_delta": source_floor,
            "supported_source_ba_delta": source_deltas,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 causal axial-attention blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "attention_architecture_seed_schedule_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    old_choose = base.choose_inner
    old_direction = base.mixer_direction
    old_architectures = base.ARCHITECTURES
    try:
        base.choose_inner = choose_inner
        base.mixer_direction = axial_direction
        base.ARCHITECTURES = ARCHITECTURES
        with contextlib.redirect_stdout(io.StringIO()):
            diagnostic, evidence, audits = base.run_nested(
                dev, champion, smoke=smoke, smoke_market=smoke_market,
            )
    finally:
        base.choose_inner = old_choose
        base.mixer_direction = old_direction
        base.ARCHITECTURES = old_architectures
    diagnostic.rename(columns={"v150_model": "v153_model"}, inplace=True)
    evidence.rename(columns={"mixer_prob": "axial_prob", "v150_model": "v153_model"}, inplace=True)
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V153 AXIAL-ATTENTION] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    old_seed = base.SEED
    try:
        base.SEED = SEED
        result = base.paired_nested_bootstrap(evidence, draws)
    finally:
        base.SEED = old_seed
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V153 causal axial-attention policy"
    return result


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
    smoke: bool,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V153 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    attention_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["axial_attention"]["static_spec"]["block_n"] == BLOCK_N
        and audit[side]["axial_attention"]["fixed_architecture_seed_and_schedule"]
        and audit[side]["axial_attention"]["duplicate_class_balanced_bce"]
        and audit[side]["axial_attention"]["factorized_horizon_then_channel_attention"]
        and audit[side]["axial_attention"]["fixed_sinusoidal_positions"]
        and not audit[side]["axial_attention"]["text_or_pretrained_transformer"]
        and not audit[side]["target_rows_used_for_robust_scale_or_training"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["v57_pretrained_sec_document_transformer"]
        and not audit[side]["v150_temporal_channel_mlp_mixer"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    native_bootstrap = candidate_summary["robustness"]["bootstrap"]
    required_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - baseline["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - baseline["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native_bootstrap),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "multichannel_causal_axial_attention_contract_verified": attention_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V153 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass,
        "material_checks": checks,
        "outer_baseline": baseline,
        "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "candidate_native_robustness_bootstrap": native_bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline,
            "candidate": canonical_candidate,
            "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original,
        "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS,
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V153 nested key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V153 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V153 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V153", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V153_DIAGNOSTIC_CAUSAL_AXIAL_ATTENTION": evaluation["candidate_summary"],
                "V153_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V153", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V153", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V153_MULTICHANNEL_CAUSAL_AXIAL_ATTENTION_REPORT.json": {
            "version": "V153", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "runtime": ACTIVE_RUNTIME,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V153", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V153_CAUSAL_AXIAL_ATTENTION_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V153_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    return {
        "output": str(out),
        "required_controller_reports": [
            "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
        ],
    }


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
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V153 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V153 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V153 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = configure_execution(bounded)
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        model = CausalAxialAttention()
        parameter_n = sum(parameter.numel() for parameter in model.parameters())
        print(json.dumps(clean({
            "status": "AUDIT_OK",
            "hypothesis": HYPOTHESIS,
            "authority": authority,
            "runtime": runtime,
            "dev_rows": len(dev),
            "v69_rows": len(champion),
            "features": FEATURES,
            "static_spec": STATIC_SPEC,
            "parameter_n": parameter_n,
            "architecture": "two residual blocks of separate horizon-axis then channel-axis multihead self-attention with fixed sinusoidal positions, mean/recent pooling and balanced BCE head",
            "causal_numeric_only": True,
            "target_row_labels_used": False,
            "micro_tuning_or_post_outer_change": False,
            "v57_pretrained_text_transformer_collision": False,
            "v150_attention_free_mlp_mixer_collision": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "full_imports_runtime_limits_and_requires_gpu0": True,
                "unavailable_gpu_fails_closed_before_output_creation": True,
                "bounded_cpu_support_gpu_model_calls": 0,
                "research_staging_v153_compatible": True,
                "required_reports": [
                    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
                ],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(
            dev, champion, smoke=bounded, smoke_market=args.smoke_market,
        )
        if args.support_probe:
            require(args.smoke_market == "KR", "V153 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({
                "status": "SUPPORT_PROBE_OK",
                "hypothesis": HYPOTHESIS,
                "runtime": runtime,
                "market": "KR",
                "fold": 2,
                "strict_inner_chronology": audit["inner_chronology"],
                "strict_outer_chronology": audit["outer_chronology"],
                "raw_inner_selected": audit["policy"]["selected"],
                "raw_inner_best_non_noop": best_non_noop,
                "raw_outer_baseline": audit["outer_baseline"],
                "raw_outer_selected": audit["outer_candidate"],
                "raw_outer_best_non_noop_evaluation_only": audit["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
                "inner_model_audit": audit["inner_model_audit"],
                "outer_model_audit": audit["outer_model_audit"],
                "selection_locked_before_outer_evaluation": True,
                "nested_bootstrap_executed": False,
                "nested_bootstrap_reason": "KR:2 support validates model support only; standard material bootstrap remains US smoke/full evaluation",
                "exact_entire_v69_fallback_contract": True,
                "output_written": False,
            }), ensure_ascii=False, indent=2))
            return
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
            smoke=args.smoke_test,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "runtime": runtime,
        "folds_executed": len(audits),
        "smoke_market": args.smoke_market if args.smoke_test else None,
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "current_material_gate": material_gate(evaluation),
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["inner_model_audit"] = audits[0]["inner_model_audit"]
        summary["outer_model_audit"] = audits[0]["outer_model_audit"]
        summary["candidate_native_robustness_bootstrap"] = evaluation["candidate_native_robustness_bootstrap"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
