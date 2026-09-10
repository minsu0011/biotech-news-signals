"""V121 deterministic hyperdimensional bind-and-bundle direction challenger.

Each same-market past train cut robustly scales the immutable 37 causal
numeric coordinates, maps each coordinate to one of nine fixed ordinal states
or an explicit missing state, and binds that state to a deterministic bipolar
feature item-memory.  The 37 bound vectors are bundled and majority-binarized
into a 1,024-dimensional vector-symbolic row code.  Duplicate rows collapse to
equal-weight event-group codes; per-class majority bundles form two fixed HDC
prototypes.  Their cosine similarities are normalized directly under equal
class priors, with no fitted head, temperature, kernel objective, or retrieval.

This is not recurrence/RQA, polynomial chaos, Gaussian density, projector
density, path dynamics, graph, tree, kernel, or neural training.  Earlier
committed V69 OOF alone chooses exact V69 or fixed .25/.50 blends in the inner
past.  Outer labels are evaluation-only under strict 35-minute nested
chronology; V69 confidence/high_conf remain exact.  Material failure restores
the exact entire atomic V69 frame.  A class-stratified event-group Bayesian
bootstrap separately audits bundled-prototype stability.
"""
from __future__ import annotations

import os

CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
if not CONTROLLER_OUTPUT:
    for _name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[_name] = "2"

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v85_recent_regime_entropy_balance as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V121_DETERMINISTIC_HYPERDIMENSIONAL_BIND_BUNDLE_V1"
VERSION = 121
HYPOTHESIS = "DETERMINISTIC_HYPERDIMENSIONAL_BIND_BUNDLE_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v85_recent_regime_entropy_balance.py"
EXPECTED_SCAFFOLD_SHA256 = "511f8c2eaab2d9ec9b58f61ce3bed0c0a1539bc5b5757fa8768d556dcad66a84"

HDC_DIMENSION = 1024
NUMERIC_LEVELS = 9
ROBUST_STATE_CLIP = 5.0
SIMILARITY_FLOOR = 1e-12
SEED = 12101
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)

ARCHITECTURES = (
    {"name": "HYPERDIMENSIONAL_BIND_BUNDLE_W0.25", "weight": 0.25},
    {"name": "HYPERDIMENSIONAL_BIND_BUNDLE_W0.50", "weight": 0.50},
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
blend_probability = scaffold.blend_probability
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44


def verify_static_spec() -> dict[str, Any]:
    require(len(FEATURES) == 37 and HDC_DIMENSION == 1024, "V121 HDC dimension changed")
    require(HDC_DIMENSION & (HDC_DIMENSION - 1) == 0, "V121 HDC dimension is not a power of two")
    require(
        abs(float(scaffold.ROBUST_CLIP) - ROBUST_STATE_CLIP) <= 1e-15,
        "V121 robust state clip mismatch",
    )
    payload = (
        "numeric:37|robust_clip:5|uniform_levels:9|missing_state:1|hdc_dim:1024|"
        "sha256_bipolar_item_memory|correlated_level_memory|multiply_bind|"
        "majority_bundle|equal_event_group|class_prototype|cosine_equal_prior"
    )
    return {
        "causal_numeric_n": len(FEATURES), "explicit_missing_state_n": 1,
        "numeric_level_n": NUMERIC_LEVELS, "hdc_dimension": HDC_DIMENSION,
        "robust_state_clip": ROBUST_STATE_CLIP,
        "item_memory": "deterministic SHA256 bipolar feature vectors",
        "level_memory": "deterministic correlated bipolar ordinal vectors",
        "binding": "elementwise bipolar multiplication",
        "bundling": "coordinatewise sum then deterministic majority sign",
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "representation_label_independent": True, "equal_class_prior": True,
        "fitted_head": False, "fitted_temperature": False,
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V85 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "generic immutable authority, chronology, metrics, robust numeric transform, blending, canonical gate, and atomic IO only; no V85 output is read",
    }
    audit["v121_access_contract"] = {
        "immutable_v36_dev_only": True,
        "atomic_output_v69_only": True,
        "failed_outputs_read": False,
        "dev_extension_read": False,
        "role_assignment_read": False,
        "research_seal_read": False,
        "final_reserve_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    process = psutil.Process()
    process.cpu_affinity(list(PREPARATION_CPU_AFFINITY))
    actual = tuple(sorted(process.cpu_affinity()))
    require(actual == PREPARATION_CPU_AFFINITY, "V121 preparation CPU affinity is not exactly 30-31")
    return {
        "requested_logical_cpus": list(PREPARATION_CPU_AFFINITY),
        "actual_logical_cpus": list(actual), "exact_match": True,
        "process_id": process.pid,
    }


def hash_bipolar(label: str) -> np.ndarray:
    blocks: list[np.ndarray] = []
    block = 0
    while sum(len(item) for item in blocks) < HDC_DIMENSION:
        digest = hashlib.sha256(f"V121|{label}|{block}".encode("utf-8")).digest()
        blocks.append(np.unpackbits(np.frombuffer(digest, dtype=np.uint8)))
        block += 1
    bits = np.concatenate(blocks)[:HDC_DIMENSION]
    return np.where(bits > 0, 1.0, -1.0)


def build_item_memory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item = np.vstack([hash_bipolar(f"feature:{name}") for name in FEATURES])
    base = hash_bipolar("ordinal-level-base")
    index = np.arange(HDC_DIMENSION, dtype=np.uint64)
    keys = (index * np.uint64(2654435761) + np.uint64(1013904223)) & np.uint64(0xFFFFFFFF)
    order = np.argsort(keys, kind="stable")
    levels = []
    for level in range(NUMERIC_LEVELS):
        vector = base.copy()
        flip_n = int(round(level * HDC_DIMENSION / float(NUMERIC_LEVELS - 1)))
        vector[order[:flip_n]] *= -1.0
        levels.append(vector)
    missing = hash_bipolar("explicit-missing-state")
    return item, np.vstack(levels), missing


ITEM_MEMORY, LEVEL_MEMORY, MISSING_MEMORY = build_item_memory()


def hdc_rows(frame: pd.DataFrame, processor: Any) -> tuple[np.ndarray, dict[str, Any]]:
    raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    missing = ~np.isfinite(raw)
    numeric = np.clip(processor.transform(frame), -ROBUST_STATE_CLIP, ROBUST_STATE_CLIP)
    level = np.floor(
        (numeric + ROBUST_STATE_CLIP) * NUMERIC_LEVELS / (2.0 * ROBUST_STATE_CLIP)
    ).astype(int)
    level = np.clip(level, 0, NUMERIC_LEVELS - 1)
    bundle = np.zeros((len(frame), HDC_DIMENSION), dtype=np.float64)
    level_counts = np.zeros(NUMERIC_LEVELS, dtype=np.int64)
    for feature_index in range(len(FEATURES)):
        state = LEVEL_MEMORY[level[:, feature_index]].copy()
        state[missing[:, feature_index]] = MISSING_MEMORY
        bundle += state * ITEM_MEMORY[feature_index]
        present = level[~missing[:, feature_index], feature_index]
        level_counts += np.bincount(present, minlength=NUMERIC_LEVELS)
    code = np.where(bundle >= 0.0, 1.0, -1.0)
    require(code.shape == (len(frame), HDC_DIMENSION), "V121 HDC row-code shape changed")
    require(np.isfinite(code).all() and np.all(np.abs(code) == 1.0), "V121 HDC code is not bipolar")
    return code, {
        "row_n": len(frame), "missing_state_count": int(missing.sum()),
        "numeric_level_counts": level_counts.tolist(),
        "positive_bit_fraction": float(np.mean(code > 0.0)),
        "row_code_sha256": array_sha256(code),
    }


def group_codes(
    frame: pd.DataFrame, code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(frame) == len(code), "V121 group-code shape mismatch")
    state = pd.DataFrame(code)
    state["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    state["y"] = frame.y.to_numpy(int)
    grouped = state.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V121 event group crosses direction labels")
    mean_code = grouped[list(range(HDC_DIMENSION))].mean().to_numpy(float)
    result = np.where(mean_code >= 0.0, 1.0, -1.0)
    target = grouped.y.first().to_numpy(int)
    group_ids = grouped.size().index.to_numpy(str)
    require(np.all(np.abs(result) == 1.0), "V121 event-group code is not bipolar")
    return result, target, group_ids


def class_prototype(code: np.ndarray, target: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    selected = code[target == label]
    require(len(selected) >= 80, f"V121 class {label} event-group support too small")
    vote = selected.mean(axis=0)
    prototype = np.where(vote >= 0.0, 1.0, -1.0) / math.sqrt(float(HDC_DIMENSION))
    require(abs(float(np.linalg.norm(prototype)) - 1.0) <= 1e-12, "V121 prototype norm changed")
    return prototype, selected, {
        "label": label, "event_group_n": len(selected),
        "vote_absolute_mean": float(np.mean(np.abs(vote))),
        "vote_zero_fraction": float(np.mean(vote == 0.0)),
        "prototype_positive_fraction": float(np.mean(prototype > 0.0)),
        "prototype_sha256": array_sha256(prototype),
    }


def cosine_probability(query_code: np.ndarray, down: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = query_code / math.sqrt(float(HDC_DIMENSION))
    down_cosine = np.clip(query @ down, -1.0, 1.0)
    up_cosine = np.clip(query @ up, -1.0, 1.0)
    down_score = 0.5 * (1.0 + down_cosine)
    up_score = 0.5 * (1.0 + up_cosine)
    probability = up_score / np.maximum(up_score + down_score, SIMILARITY_FLOOR)
    require(np.isfinite(probability).all(), "V121 HDC cosine probability invalid")
    return probability, down_cosine, up_cosine


def native_prototype_bootstrap(
    query_code: np.ndarray,
    down_code: np.ndarray,
    up_code: np.ndarray,
    nominal_probability: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    require(draws >= 16, "V121 native prototype bootstrap draw count too small")
    generator = np.random.default_rng(seed)
    down_weight = generator.exponential(size=(len(down_code), draws))
    up_weight = generator.exponential(size=(len(up_code), draws))
    down_weight /= down_weight.sum(axis=0, keepdims=True)
    up_weight /= up_weight.sum(axis=0, keepdims=True)
    down_prototype = np.where(down_code.T @ down_weight >= 0.0, 1.0, -1.0) / math.sqrt(float(HDC_DIMENSION))
    up_prototype = np.where(up_code.T @ up_weight >= 0.0, 1.0, -1.0) / math.sqrt(float(HDC_DIMENSION))
    query = query_code / math.sqrt(float(HDC_DIMENSION))
    down_score = 0.5 * (1.0 + np.clip(query @ down_prototype, -1.0, 1.0))
    up_score = 0.5 * (1.0 + np.clip(query @ up_prototype, -1.0, 1.0))
    probability = up_score / np.maximum(up_score + down_score, SIMILARITY_FLOOR)
    require(probability.shape == (len(query_code), draws) and np.isfinite(probability).all(), "V121 native bootstrap invalid")
    median_probability = np.median(probability, axis=1)
    row_std = np.std(probability, axis=1)
    direction_agreement = np.mean((probability >= 0.5) == (nominal_probability[:, None] >= 0.5))
    quantiles = (0.10, 0.50, 0.90)
    return {
        "contract": "class-stratified event-group Bayesian bootstrap of bipolar HDC prototypes",
        "draws": draws, "seed": seed, "class_stratified": True,
        "event_group_unit_weights": True, "target_labels_used": False,
        "hdc_dimension": HDC_DIMENSION,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(probability - nominal_probability[:, None]))),
        "median_probability_mean_absolute_deviation": float(np.mean(np.abs(median_probability - nominal_probability))),
        "direction_agreement_with_nominal": float(direction_agreement),
        "row_probability_std_quantiles": {str(q): float(np.quantile(row_std, q)) for q in quantiles},
        "median_probability_sha256": array_sha256(median_probability),
    }


def hyperdimensional_bind_bundle_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
    native_draws: int = 0,
    native_seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V121 expects one market per fit")
    require(str(train.market.iloc[0]) == str(target_frame.market.iloc[0]), "V121 market fit mismatch")
    processor = scaffold.RobustNumericState().fit(train)
    train_row_code, train_encoding_audit = hdc_rows(train, processor)
    train_code, train_y, train_groups = group_codes(train, train_row_code)
    target_code, target_encoding_audit = hdc_rows(target_frame, processor)
    down_prototype, down_code, down_audit = class_prototype(train_code, train_y, 0)
    up_prototype, up_code, up_audit = class_prototype(train_code, train_y, 1)
    probability, down_cosine, up_cosine = cosine_probability(target_code, down_prototype, up_prototype)
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    native = None if native_draws <= 0 else native_prototype_bootstrap(
        target_code, down_code, up_code, probability, native_draws, native_seed,
    )
    quantiles = (0.10, 0.50, 0.90)
    return probability, {
        "market": str(train.market.iloc[0]), "train_n": len(train), "target_n": len(target_frame),
        "causal_numeric_feature_n": len(FEATURES), "explicit_missing_state_n": 1,
        "hdc_dimension": HDC_DIMENSION, "numeric_level_n": NUMERIC_LEVELS,
        "causal_pre_event_only": True,
        "categorical_feature_n": 0, "text_feature_n": 0,
        "source_or_ticker_identity": False,
        "architecture": "deterministic hyperdimensional item-level binding, row bundling, and duplicate-balanced class prototypes",
        "static_spec": STATIC_SPEC, "train_scaling_only": True,
        "event_group_equal_weight": True, "train_event_group_n": len(train_groups),
        "train_encoding": train_encoding_audit, "target_encoding": target_encoding_audit,
        "down_prototype": down_audit, "up_prototype": up_audit,
        "prototype_unit_norm": True, "equal_class_prior": True,
        "deterministic_item_memory": True, "correlated_ordinal_level_memory": True,
        "bipolar_multiplicative_binding": True, "majority_bundling": True,
        "inverse_or_determinant": False, "fitted_temperature": False,
        "fitted_direction_coefficient_n": 0, "learned_discriminative_head": False,
        "recurrence_rqa_or_polynomial_chaos": False,
        "gaussian_copula_qda_vmf_or_quantum_density": False,
        "path_signature_spd_or_dynamic_operator": False,
        "graph_tree_kernel_objective_or_neural": False,
        "nearest_analog_or_retrieval": False,
        "target_rows_used_for_scaling_quantization_or_prototype": False,
        "target_labels_used": False,
        "native_prototype_bootstrap": native,
        "prototype_cosine": float(down_prototype @ up_prototype),
        "down_cosine_quantiles": {str(q): float(np.quantile(down_cosine, q)) for q in quantiles},
        "up_cosine_quantiles": {str(q): float(np.quantile(up_cosine, q)) for q in quantiles},
        "probability_quantiles": {str(q): float(np.quantile(probability, q)) for q in quantiles},
        "probability_sha256": array_sha256(probability),
    }


def inner_partition(
    dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - scaffold.EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} OOF")
    split = int(math.floor(len(prior) * (1.0 - scaffold.INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, market, inner_start, inner_valid)
    require(len(inner_train) >= (700 if market == "US" else 250), "inner V121 training cut too small")
    audit = chronology(inner_train, inner_valid, f"V121 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(
    inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = hyperdimensional_bind_bundle_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 10.0 * baseline_metric["all_trade_mean_signed_net"]),
    }]
    model_eligible = bool(
        model_audit["train_scaling_only"]
        and model_audit["event_group_equal_weight"]
        and model_audit["static_spec"]["representation_label_independent"]
        and model_audit["static_spec"]["hdc_dimension"] == HDC_DIMENSION
        and model_audit["down_prototype"]["event_group_n"] >= 80
        and model_audit["up_prototype"]["event_group_n"] >= 80
        and model_audit["prototype_unit_norm"]
        and model_audit["deterministic_item_memory"]
        and model_audit["bipolar_multiplicative_binding"]
        and model_audit["majority_bundling"]
        and model_audit["fitted_direction_coefficient_n"] == 0
        and not model_audit["learned_discriminative_head"]
        and not model_audit["target_rows_used_for_scaling_quantization_or_prototype"]
        and not model_audit["target_labels_used"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": values,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta, "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"] + 10.0 * values["all_trade_mean_signed_net"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (
            trial["score"], trial["auc_delta"], trial["all_trade_net_delta"],
            trial["name"] == "V69_NOOP",
        ),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 hyperdimensional-bind-bundle logit blend; fixed score and BA/net guards",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v121_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([('US', 2)] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V121 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(
            dev, champion, market, outer_start,
        )
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = hyperdimensional_bind_bundle_direction(
            outer_train, outer_valid,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
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
        diagnostic.loc[positions, "v121_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid, blend_probability(baseline, challenger, architecture["weight"]),
                confidence, high,
            ) for architecture in ARCHITECTURES
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["hyperdimensional_bind_bundle_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v121_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": baseline_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V121 HDC BIND BUNDLE] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V121 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V121 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V121 changed V69 high-confidence")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V121 nested-bootstrap blocks")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    net_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        target = sample.y.to_numpy(int)
        if np.unique(target).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(target, candidate) - roc_auc_score(target, baseline)))
        ba_delta.append(float(
            balanced_accuracy_score(target, candidate >= 0.5)
            - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        net_delta.append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))
    require(len(auc_delta) >= int(0.90 * draws), "V121 nested bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "contract_key": "selected.material_gate.nested_bootstrap",
        "standardized": True,
        "method": "standardized paired market-fold-time block bootstrap over inner-locked V121 policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta), "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(
    champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame,
    audits: list[dict[str, Any]], draws: int, smoke: bool,
) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V121/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    hdc_valid = all(
        audit["outer_model_audit"]["causal_pre_event_only"]
        and audit["outer_model_audit"]["train_scaling_only"]
        and audit["outer_model_audit"]["event_group_equal_weight"]
        and audit["outer_model_audit"]["static_spec"]["representation_label_independent"]
        and audit["outer_model_audit"]["hdc_dimension"] == HDC_DIMENSION
        and audit["outer_model_audit"]["down_prototype"]["event_group_n"] >= 80
        and audit["outer_model_audit"]["up_prototype"]["event_group_n"] >= 80
        and audit["outer_model_audit"]["prototype_unit_norm"]
        and audit["outer_model_audit"]["deterministic_item_memory"]
        and audit["outer_model_audit"]["correlated_ordinal_level_memory"]
        and audit["outer_model_audit"]["bipolar_multiplicative_binding"]
        and audit["outer_model_audit"]["majority_bundling"]
        and audit["outer_model_audit"]["equal_class_prior"]
        and not audit["outer_model_audit"]["inverse_or_determinant"]
        and audit["outer_model_audit"]["fitted_direction_coefficient_n"] == 0
        and not audit["outer_model_audit"]["learned_discriminative_head"]
        and not audit["outer_model_audit"]["target_rows_used_for_scaling_quantization_or_prototype"]
        and not audit["outer_model_audit"]["target_labels_used"]
        for audit in audits
    )
    controller_native_bootstrap = candidate_summary["robustness"]["bootstrap"]
    required_controller_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
    model_native_bootstrap = [audit["outer_model_audit"]["native_prototype_bootstrap"] for audit in audits]
    model_native_valid = all(
        item is not None
        and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["class_stratified"]
        and item["event_group_unit_weights"]
        and not item["target_labels_used"]
        and 0.0 <= item["direction_agreement_with_nominal"] <= 1.0
        for item in model_native_bootstrap
    )
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "controller_native_robustness_bootstrap_keys": required_controller_native.issubset(controller_native_bootstrap),
        "hdc_prototype_native_bootstrap_contract_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "deterministic_hyperdimensional_bind_bundle_contract_verified": hdc_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V121 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "nested_bootstrap": nested_bootstrap,
        "controller_native_robustness_bootstrap": controller_native_bootstrap,
        "model_native_prototype_bootstrap": model_native_bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "immutability": {
            "confidence_exact": confidence_exact,
            "baseline_confidence_sha256": array_sha256(champion.confidence_signal.to_numpy(float)),
            "candidate_confidence_sha256": array_sha256(diagnostic_original.confidence_signal.to_numpy(float)),
            "baseline_highconf_sha256": array_sha256(champion.high_conf.to_numpy(bool)),
            "candidate_highconf_sha256": array_sha256(diagnostic_original.high_conf.to_numpy(bool)),
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]), "material_pass": evaluation["material_pass"],
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_prototype_bootstrap": evaluation["model_native_prototype_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(
        gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap",
        "V121 nested bootstrap contract key mismatch",
    )
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V121 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V121 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / f"V{VERSION}").resolve(), "V121 output is outside controller research/staging/V121")
    require(out.exists() and out.is_dir(), "V121 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V121 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V121 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing),
        "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path, authority: dict[str, Any], evidence: pd.DataFrame,
    audits: list[dict[str, Any]], evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V121 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V121 output is not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V121 selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V121", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V121_DIAGNOSTIC_HYPERDIMENSIONAL_BIND_BUNDLE": evaluation["candidate_summary"],
                "V121_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V121", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected,
            "candidate": evaluation["candidate_summary"], "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V121", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "all_trade_net_delta": evaluation["outer_net_delta"],
            },
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "HYPERDIMENSIONAL_BIND_BUNDLE_REPORT.json": {
            "version": "V121", "hypothesis": HYPOTHESIS,
            "features": FEATURES,
            "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V121", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap",
            "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V121", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
            "model_native_hdc_prototype_bootstrap": evaluation["model_native_prototype_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V121", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if selected["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit, "completed_at": scaffold.now(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(pd.DataFrame([{
        "market": audit["market"], "fold": audit["fold"],
        "selected_model": audit["policy"]["selected"]["name"],
        "outer_auc_delta": audit["outer_auc_delta"],
        "outer_ba_delta": audit["outer_ba_delta"],
        "outer_net_delta": audit["outer_net_delta"],
        "train_event_group_n": audit["outer_model_audit"]["train_event_group_n"],
        "down_event_group_n": audit["outer_model_audit"]["down_prototype"]["event_group_n"],
        "up_event_group_n": audit["outer_model_audit"]["up_prototype"]["event_group_n"],
        "hdc_dimension": audit["outer_model_audit"]["hdc_dimension"],
        "native_bootstrap_draws": audit["outer_model_audit"]["native_prototype_bootstrap"]["draws"],
        "strict_outer_35m_embargo": audit["outer_chronology"]["strict_35m_embargo"],
        "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V121_HYPERDIMENSIONAL_BIND_BUNDLE_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V121_HYPERDIMENSIONAL_BIND_BUNDLE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V121_DIAGNOSTIC_HYPERDIMENSIONAL_BIND_BUNDLE_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V121_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({
        "version": VERSION, "hypothesis": HYPOTHESIS,
        "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files,
    }, out / "ARTIFACT_MANIFEST.json")
    return {
        "output": str(out), "artifact_count": len(files) + 1,
        "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode cannot accompany controller output",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V121 full run requires MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    affinity_audit = preparation_affinity() if (args.audit_only or args.smoke_test) else {
        "requested_logical_cpus": None, "actual_logical_cpus": psutil.Process().cpu_affinity(),
        "exact_match": None, "reason": "controller-authorized full run is not preparation",
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "deterministic hyperdimensional bind-and-bundle with duplicate-balanced class prototypes",
            "hdc_dimension": STATIC_SPEC["hdc_dimension"],
            "numeric_levels": STATIC_SPEC["numeric_level_n"],
            "causal_numeric_only": True, "categorical_feature_n": 0, "text_feature_n": 0,
            "source_or_ticker_identity": False,
            "controller_no_arg_contract": {
                "environment_variable": "MARKET_BIO_VERSION_OUTPUT",
                "implicit_mode": "full_run", "output_bound_to_environment": True,
                "required_reports": [
                    "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json",
                ],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True,
                "controller_native_robustness_bootstrap": True,
                "model_native_hdc_prototype_bootstrap": True,
                "exact_entire_v69_fallback": True,
                "unexpected_output_conflict_fail_closed": True,
                "authorized_controller_prefiles_required": True,
            },
            "canonical_controller_gate_checks": 14,
            "resource_policy": {
                "smoke_fold": "US:2", "smoke_cpu_thread_cap": SMOKE_THREADS,
                "gpu_model_calls": 0, "cpu_affinity": affinity_audit,
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    limit = SMOKE_THREADS if args.smoke_test else None
    with threadpool_limits(limits=limit):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion, diagnostic, evidence, audits,
            SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS,
            smoke=args.smoke_test,
        )
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["architecture"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "fold_keys": [f"{audit['market']}:{audit['fold']}" for audit in audits],
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "inner_baseline": audits[0]["policy"]["baseline"],
        "inner_selected": audits[0]["policy"]["selected"],
        "inner_best_non_noop": best_non_noop,
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_evaluation_only_architectures": audits[0]["outer_architecture_diagnostics_evaluation_only"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "model_audit": audits[0]["outer_model_audit"],
        "current_material_gate": current_material_gate(evaluation),
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "resource_evidence": {
            "threadpool_limit": limit, "gpu_model_calls": 0,
            "cpu_affinity": affinity_audit, "threadpools": threadpool_info(),
        },
        "output_written": False,
    }
    if args.smoke_test:
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    output = write_outputs(args.output, authority, evidence, audits, evaluation)
    summary["output_written"] = True
    summary["output"] = output
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
