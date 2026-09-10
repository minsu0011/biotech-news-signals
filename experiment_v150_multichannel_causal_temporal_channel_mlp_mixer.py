"""V150 causal temporal/channel MLP-Mixer direction challenger.

Every embargoed same-market train cut robust-scales the immutable 37 causal
numeric fields and maps them into the established eight semantic channels at
eight observed pre-event horizons, ordered from 120 minutes to 1 minute before
the event.  A fixed tiny two-block MLP-Mixer alternates a shared horizon-token
MLP with a shared semantic-channel MLP, followed by mean-plus-recent pooling and
one direction head.  Training uses fixed duplicate/class-balanced BCE, seed and
schedule.  The learned map contains no convolution, recurrence or attention.

The original depthwise-TCN proposal was rejected before implementation because
V55 already implements a dilated residual temporal CNN.  This replacement is
also distinct from V52's static embedding MLP, V82's masked-state self-supervised
encoder, V89's explicit low-rank bilinear state model and V132's fixed recurrent
echo reservoir.  Inner-past evidence selects exact V69 or fixed .25/.50 blends;
outer labels are evaluation-only after a strict 35-minute embargo/event purge.
V69 confidence/high-confidence stay exact, and any current material-gate failure
restores the exact entire atomic V69 DataFrame.

Audit, US:2 smoke and KR:2 support use CPU30-31/two threads and make zero GPU
model calls.  Full execution is accepted only through no-argument
MARKET_BIO_VERSION_OUTPUT, verifies runtime_limits GPU0 authority and fails
closed before output creation if CUDA GPU0 is unavailable.
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
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from threadpoolctl import threadpool_limits

import experiment_v144_multichannel_weak_ordinal_motif_transition_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V150_TEMPORAL_CHANNEL_MLP_MIXER_V1"
VERSION = 150
HYPOTHESIS = "MULTICHANNEL_CAUSAL_TEMPORAL_CHANNEL_MLP_MIXER_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v144_multichannel_weak_ordinal_motif_transition_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "19817afa20b5243fea7831fd1d26a17cc2159085c5a774a2d9524a82d0f4064c"

CHANNEL_N = 8
HORIZON_N = 8
BLOCK_N = 2
TOKEN_HIDDEN_N = 16
CHANNEL_HIDDEN_N = 32
POOL_N = 2 * CHANNEL_N
SEED = 15001
FULL_EPOCHS = 40
BOUNDED_EPOCHS = 6
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-3
GRADIENT_CLIP = 2.0
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "TEMPORAL_CHANNEL_MLP_MIXER_W0.25", "weight": 0.25},
    {"name": "TEMPORAL_CHANNEL_MLP_MIXER_W0.50", "weight": 0.50},
)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
chronology = scaffold.chronology
aligned_dev = scaffold.aligned_dev
prior_train = scaffold.prior_train
inner_partition = scaffold.inner_partition
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44
numeric = scaffold.numeric

ACTIVE_DEVICE = torch.device("cpu")
ACTIVE_EPOCHS = BOUNDED_EPOCHS
ACTIVE_RUNTIME: dict[str, Any] = {
    "mode": "unconfigured",
    "device": "cpu",
    "gpu_model_calls": 0,
}


STATIC_SPEC = {
    "channel_n": CHANNEL_N,
    "channel_names": list(scaffold.STATIC_SPEC["channel_names"]),
    "horizon_n": HORIZON_N,
    "path_order": "120m,60m,30m,15m,10m,5m,2m,1m past-to-recent",
    "artificial_origin_or_time_coordinate": False,
    "block_n": BLOCK_N,
    "token_mlp": [HORIZON_N, TOKEN_HIDDEN_N, HORIZON_N],
    "channel_mlp": [CHANNEL_N, CHANNEL_HIDDEN_N, CHANNEL_N],
    "residual_layer_norm_gelu": True,
    "pool": "concatenate all-horizon mean and nearest-pre-event token",
    "head": f"linear {POOL_N} to one logit",
    "loss": "duplicate/class-balanced weighted BCEWithLogits",
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip": GRADIENT_CLIP,
    "full_epochs": FULL_EPOCHS,
    "bounded_support_epochs": BOUNDED_EPOCHS,
    "seed": SEED,
    "convolution": False,
    "recurrence": False,
    "attention": False,
    "self_supervised_pretraining": False,
    "architecture_or_schedule_grid": False,
}


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V144 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, causal path interpolation, metrics, canonical gate, blending and atomic IO only; no V144 output is read",
    }
    audit["v150_access_contract"] = {
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


class MixerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(CHANNEL_N)
        self.token_mlp = nn.Sequential(
            nn.Linear(HORIZON_N, TOKEN_HIDDEN_N),
            nn.GELU(),
            nn.Linear(TOKEN_HIDDEN_N, HORIZON_N),
        )
        self.channel_norm = nn.LayerNorm(CHANNEL_N)
        self.channel_mlp = nn.Sequential(
            nn.Linear(CHANNEL_N, CHANNEL_HIDDEN_N),
            nn.GELU(),
            nn.Linear(CHANNEL_HIDDEN_N, CHANNEL_N),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        token = self.token_norm(tensor).transpose(1, 2)
        tensor = tensor + self.token_mlp(token).transpose(1, 2)
        tensor = tensor + self.channel_mlp(self.channel_norm(tensor))
        return tensor


class TemporalChannelMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MixerBlock() for _ in range(BLOCK_N)])
        self.final_norm = nn.LayerNorm(CHANNEL_N)
        self.head = nn.Linear(POOL_N, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tensor = block(tensor)
        tensor = self.final_norm(tensor)
        pooled = torch.cat([tensor.mean(dim=1), tensor[:, -1, :]], dim=1)
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
    require(runtime_limits.GPU_ENABLED is True, "V150 runtime_limits GPU authority disabled")
    require(status["cuda_visible_devices"] == "0", "V150 runtime_limits did not bind GPU0")
    require(torch.cuda.is_available(), "V150 CUDA GPU0 unavailable under runtime_limits authority")
    require(torch.cuda.device_count() >= 1, "V150 CUDA GPU0 missing")
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


def observed_path(robust_numeric: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    path, inherited = scaffold.scaffold.observed_past_to_recent_path(robust_numeric)
    require(path.shape[1:] == (HORIZON_N, CHANNEL_N), "V150 observed path shape changed")
    return np.ascontiguousarray(path, dtype=np.float32), {
        **inherited,
        "rows": len(path),
        "shape": list(path.shape),
        "oldest_to_nearest_pre_event": True,
        "artificial_origin_removed": True,
        "time_coordinate_removed": True,
        "path_sha256": array_sha256(path),
    }


def set_model_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if ACTIVE_DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mixer_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V150 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_path, train_path_audit = observed_path(train_state[:, :len(FEATURES)])
    target_path, target_path_audit = observed_path(target_state[:, :len(FEATURES)])
    target = ordered.y.to_numpy(np.float32)
    sample_weight = numeric.duplicate_class_weights(ordered).astype(np.float32)
    sample_weight = sample_weight / float(np.mean(sample_weight))
    require(np.unique(target).size == 2 and np.isfinite(sample_weight).all(), "V150 weighted target invalid")

    set_model_seed(SEED)
    model = TemporalChannelMixer().to(ACTIVE_DEVICE)
    parameter_n = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
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
        require(bool(torch.isfinite(loss).item()), "V150 non-finite training loss")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        require(bool(torch.isfinite(gradient_norm).item()), "V150 non-finite gradient")
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.from_numpy(target_path).to(ACTIVE_DEVICE))).cpu().numpy()
    probability = np.clip(probability.astype(float), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V150 probability invalid")
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
        "mixer": {
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
            "convolution": False,
            "recurrence": False,
            "attention": False,
            "self_supervised_pretraining": False,
        },
        "target_rows_used_for_robust_scale_or_training": False,
        "target_labels_used": False,
        "v55_dilated_residual_temporal_cnn": False,
        "v52_static_embedding_mlp": False,
        "v82_masked_state_self_supervised_encoder": False,
        "v89_field_aware_low_rank_bilinear": False,
        "v132_fixed_echo_state_reservoir": False,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def supported_source_ba_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, dict[str, float]]:
    deltas: dict[str, float] = {}
    for source, positions in frame.groupby("source_family", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        target = frame.iloc[index].y.to_numpy(int)
        if len(index) < 40 or np.unique(target).size != 2:
            continue
        deltas[str(source)] = float(
            balanced_accuracy_score(target, candidate[index] >= 0.5)
            - balanced_accuracy_score(target, baseline[index] >= 0.5)
        )
    return (min(deltas.values()) if deltas else 0.0), deltas


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = mixer_direction(inner_train, inner_valid)
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
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 temporal/channel MLP-Mixer blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "mixer_architecture_seed_schedule_blend_or_source_floor_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v150_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([(smoke_market, 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V150 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = mixer_direction(outer_train, outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else blend_probability(
            baseline, challenger, selected["architecture"]["weight"],
        )
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v150_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid,
                blend_probability(baseline, challenger, architecture["weight"]),
                confidence,
                high,
            ) for architecture in ARCHITECTURES
        }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["mixer_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v150_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market,
            "fold": fold,
            "inner_chronology": inner_chronology,
            "outer_chronology": outer_chronology,
            "policy": policy,
            "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric,
            "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V150 MLP-MIXER] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V150 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V150 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V150 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V150 nested-bootstrap blocks")
    generator = np.random.default_rng(SEED)
    values = {"auc_delta": [], "balanced_accuracy_delta": [], "all_trade_net_delta": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        values["auc_delta"].append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        values["balanced_accuracy_delta"].append(float(
            balanced_accuracy_score(target, candidate >= 0.5)
            - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V150 bootstrap lost too many draws")

    def interval(items: list[float]) -> dict[str, Any]:
        array = np.asarray(items, float)
        return {
            "effective_draws": len(array),
            "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)),
            "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "method": "paired market-fold-time block bootstrap over inner-locked V150 temporal/channel MLP-Mixer policy",
        "seed": SEED,
        "requested_draws": draws,
        "blocks": len(blocks),
        **{name: interval(items) for name, items in values.items()},
    }


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
    require(candidate_summary["research_gate"] == canonical_candidate, "V150 canonical mismatch")
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
    mixer_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["mixer"]["static_spec"]["block_n"] == BLOCK_N
        and audit[side]["mixer"]["fixed_architecture_seed_and_schedule"]
        and audit[side]["mixer"]["duplicate_class_balanced_bce"]
        and not audit[side]["mixer"]["convolution"]
        and not audit[side]["mixer"]["recurrence"]
        and not audit[side]["mixer"]["attention"]
        and not audit[side]["mixer"]["self_supervised_pretraining"]
        and not audit[side]["target_rows_used_for_robust_scale_or_training"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["v55_dilated_residual_temporal_cnn"]
        and not audit[side]["v52_static_embedding_mlp"]
        and not audit[side]["v82_masked_state_self_supervised_encoder"]
        and not audit[side]["v89_field_aware_low_rank_bilinear"]
        and not audit[side]["v132_fixed_echo_state_reservoir"]
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
        "temporal_channel_mlp_mixer_contract_verified": mixer_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V150 fallback is not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V150 nested key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V150 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V150 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V150", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V150_DIAGNOSTIC_TEMPORAL_CHANNEL_MLP_MIXER": evaluation["candidate_summary"],
                "V150_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V150", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V150", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V150_TEMPORAL_CHANNEL_MLP_MIXER_REPORT.json": {
            "version": "V150", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "runtime": ACTIVE_RUNTIME,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V150", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V150_TEMPORAL_CHANNEL_MLP_MIXER_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V150_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V150 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V150 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V150 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = configure_execution(bounded)
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        model = TemporalChannelMixer()
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
            "architecture": "two residual alternating horizon-token and semantic-channel MLP Mixer blocks plus mean/recent pooling and balanced BCE head",
            "causal_numeric_only": True,
            "target_row_labels_used": False,
            "micro_tuning_or_post_outer_change": False,
            "rejected_original_depthwise_tcn_due_v55_collision": True,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "full_imports_runtime_limits_and_requires_gpu0": True,
                "unavailable_gpu_fails_closed_before_output_creation": True,
                "bounded_cpu_support_gpu_model_calls": 0,
                "research_staging_v150_compatible": True,
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
            require(args.smoke_market == "KR", "V150 support probe is reserved for KR:2")
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
