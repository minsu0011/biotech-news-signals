"""V140 source-market-time random-effects meta-logit direction challenger.

Each strict-past same-market cut robustly scales the immutable 37 causal numeric
fields plus missing flags and collapses duplicates to equal event-group state
centroids.  Fixed source-family by chronological-tercile environments receive
separate balanced ridge-logit fits.  Per-coordinate diagonal observed-Fisher
variances feed an exact DerSimonian-Laird random-effects meta-analysis; Cochran-Q
heterogeneity continuously shrinks unstable coordinates.  The resulting single
global affine direction does not route target rows by source and uses no V69
probability, confidence, return, pairwise loss, class-tail loss, or outer label.

Inner-past labels select exact V69 or fixed .25/.50 logit blends.  Outer labels
are evaluation-only, V69 confidence/high_conf stay exact, and any material or
execution failure restores the exact entire atomic V69 frame.  Audit/support/
smoke are CPU30-31, two-thread, no-GPU, and no-write; full execution requires an
exact controller MARKET_BIO_VERSION_OUTPUT binding.
"""
from __future__ import annotations

import os

_thread_default = "32" if os.environ.get("MARKET_BIO_VERSION_OUTPUT") else "2"
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = _thread_default

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v136_duplicate_balanced_worst_class_superquantile_direction as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = ROOT / "research" / "staging" / "V140" / "LOCAL_FORBIDDEN"

VERSION = 140
HYPOTHESIS = "SOURCE_MARKET_TIME_RANDOM_EFFECTS_META_LOGIT_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v136_duplicate_balanced_worst_class_superquantile_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "7e9c3cd9fac2cbd7db23fbab68adfcac09020c4d3c6a041dda15fb5f846faef7"

STATE_DIMENSION = 74
CHRONOLOGICAL_TERCILES = 3
MIN_ENVIRONMENT_ROWS = 60
MIN_ENVIRONMENT_CLASS_ROWS = 15
RIDGE_L2 = 0.05
OPTIMIZER_MAX_ITERATIONS = 400
SCORE_CLIP = 20.0
SEED = 14001
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 96
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 24
SMOKE_THREADS = 2

ARCHITECTURES = (
    {"name": "RANDOM_EFFECTS_META_LOGIT_W0.25", "weight": 0.25},
    {"name": "RANDOM_EFFECTS_META_LOGIT_W0.50", "weight": 0.50},
)
STATIC_SPEC = {
    "numeric_feature_n": 37,
    "missing_flag_n": 37,
    "state_dimension": STATE_DIMENSION,
    "duplicate_unit": "equal event-group robust-state centroid",
    "environment": "same-market source_family x chronological training tercile",
    "environment_head": "fixed duplicate/class-balanced ridge logistic",
    "environment_head_L2": RIDGE_L2,
    "head_uncertainty": "diagonal observed Fisher standard error",
    "meta_analysis": "coordinatewise DerSimonian-Laird random effects",
    "heterogeneity": "Cochran-Q and continuous max(0,1-I2) coefficient shrink",
    "target_score": "one global environment-agnostic affine meta direction",
    "probability": "direct sigmoid with no calibration or threshold tuning",
}

require = scaffold.require
sha256 = scaffold.sha256
clean = scaffold.clean
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
bool_series = scaffold.bool_series
aligned_dev = scaffold.aligned_dev
prior_train = scaffold.prior_train
chronology = scaffold.chronology
inner_partition = scaffold.inner_partition
metric = scaffold.metric
blend_probability = scaffold.blend_probability
array_sha256 = scaffold.array_sha256
numeric = scaffold.numeric
v44 = scaffold.v44
controller = scaffold.controller


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V136 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust state, metrics, canonical gate, blending, bootstrap and atomic IO only; no V136 output is read",
    }
    audit["v140_access_contract"] = {
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


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    return scaffold.load_authorized()


def group_state_and_environment(
    frame: pd.DataFrame, state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require(state.shape == (len(frame), STATE_DIMENSION), "V140 robust state shape changed")
    work = pd.DataFrame(state)
    work["event_group_id"] = frame.event_group_id.astype(str).to_numpy()
    work["y"] = frame.y.to_numpy(int)
    work["source_family"] = frame.source_family.fillna("UNKNOWN").astype(str).to_numpy()
    work["market"] = frame.market.astype(str).to_numpy()
    work["event_time_utc"] = pd.to_datetime(frame.event_time_utc, utc=True).to_numpy()
    grouped = work.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V140 event group crosses direction labels")
    require(int(grouped.source_family.nunique().max()) == 1, "V140 event group crosses source families")
    require(int(grouped.market.nunique().max()) == 1, "V140 event group crosses markets")
    centroids = grouped[list(range(STATE_DIMENSION))].mean().to_numpy(float)
    target = grouped.y.first().to_numpy(int)
    group_ids = grouped.size().index.to_numpy(str)
    metadata = grouped[["source_family", "market", "event_time_utc"]].first()
    timestamp = pd.to_datetime(metadata.event_time_utc, utc=True)
    chronological_order = np.lexsort((group_ids, timestamp.astype("int64").to_numpy()))
    tercile = np.empty(len(group_ids), int)
    for index, positions in enumerate(np.array_split(chronological_order, CHRONOLOGICAL_TERCILES)):
        tercile[positions] = index
    source = metadata.source_family.to_numpy(str)
    market = metadata.market.to_numpy(str)
    environment = np.asarray([
        f"{market[index]}|{source[index]}|T{tercile[index] + 1}"
        for index in range(len(group_ids))
    ], dtype=object)
    require(np.isfinite(centroids).all(), "V140 event-group centroids invalid")
    return centroids, target, group_ids, environment, {
        "event_group_n": len(group_ids),
        "chronological_terciles": CHRONOLOGICAL_TERCILES,
        "raw_environment_n": int(pd.Series(environment).nunique()),
        "source_family_n": int(pd.Series(source).nunique()),
        "environment_counts": pd.Series(environment).value_counts().sort_index().astype(int).to_dict(),
    }


def balanced_event_weights(target: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    target = np.asarray(target, int)
    if extra is None:
        extra = np.ones(len(target), float)
    extra = np.asarray(extra, float)
    require(extra.shape == (len(target),) and np.isfinite(extra).all() and np.all(extra > 0.0), "V140 event weights invalid")
    weight = np.zeros(len(target), float)
    for direction in (0, 1):
        mask = target == direction
        require(int(mask.sum()) >= MIN_ENVIRONMENT_CLASS_ROWS, "V140 environment class support too small")
        weight[mask] = extra[mask] / float(extra[mask].sum())
    weight *= len(target) / float(weight.sum())
    return weight


def fit_environment_logit(
    state: np.ndarray, target: np.ndarray, extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    require(state.shape[1] == STATE_DIMENSION, "V140 environment state dimension changed")
    weights = balanced_event_weights(target, extra)
    signed = 2.0 * target.astype(float) - 1.0
    scale = float(weights.sum())

    def objective(parameters: np.ndarray) -> float:
        coefficient = parameters[:STATE_DIMENSION]
        score = state @ coefficient + parameters[-1]
        return float(np.sum(weights * np.logaddexp(0.0, -signed * score)) / scale + 0.5 * RIDGE_L2 * (coefficient @ coefficient))

    def gradient(parameters: np.ndarray) -> np.ndarray:
        coefficient = parameters[:STATE_DIMENSION]
        score = state @ coefficient + parameters[-1]
        loss_score = -signed * expit(-signed * score)
        multiplier = weights * loss_score / scale
        result = np.empty(STATE_DIMENSION + 1, float)
        result[:STATE_DIMENSION] = state.T @ multiplier + RIDGE_L2 * coefficient
        result[-1] = float(multiplier.sum())
        return result

    result = minimize(
        objective, np.zeros(STATE_DIMENSION + 1, float), jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": OPTIMIZER_MAX_ITERATIONS, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    )
    parameters = np.asarray(result.x, float)
    require(bool(result.success) and np.isfinite(parameters).all(), f"V140 environment logistic failed: {result.message}")
    probability = expit(state @ parameters[:STATE_DIMENSION] + parameters[-1])
    curvature = weights * probability * (1.0 - probability)
    information = np.empty(STATE_DIMENSION + 1, float)
    information[:STATE_DIMENSION] = np.sum(curvature[:, None] * state * state, axis=0) + RIDGE_L2 * scale
    information[-1] = float(curvature.sum())
    variance = 1.0 / np.maximum(information, 1e-8)
    return parameters, variance, {
        "rows": len(target),
        "down_up_support": [int(np.sum(target == 0)), int(np.sum(target == 1))],
        "optimizer_success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "coefficient_l2": float(np.linalg.norm(parameters[:STATE_DIMENSION])),
        "variance_q10_q50_q90": [float(np.quantile(variance, q)) for q in (0.1, 0.5, 0.9)],
    }


def random_effects_meta(
    estimates: np.ndarray, variances: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    require(estimates.shape == variances.shape and estimates.ndim == 2, "V140 meta input shape invalid")
    environment_n, parameter_n = estimates.shape
    require(environment_n >= 3 and parameter_n == STATE_DIMENSION + 1, "V140 meta support invalid")
    fixed_weight = 1.0 / np.maximum(variances, 1e-10)
    fixed_total = fixed_weight.sum(axis=0)
    fixed_mean = (fixed_weight * estimates).sum(axis=0) / fixed_total
    q = (fixed_weight * np.square(estimates - fixed_mean)).sum(axis=0)
    degrees = float(environment_n - 1)
    denominator = fixed_total - np.square(fixed_weight).sum(axis=0) / fixed_total
    tau2 = np.maximum(0.0, (q - degrees) / np.maximum(denominator, 1e-12))
    random_weight = 1.0 / (variances + tau2[None, :])
    random_mean = (random_weight * estimates).sum(axis=0) / random_weight.sum(axis=0)
    i2 = np.where(q > degrees, np.maximum(0.0, (q - degrees) / np.maximum(q, 1e-12)), 0.0)
    stability = np.clip(1.0 - i2, 0.0, 1.0)
    meta = random_mean * stability
    require(np.isfinite(meta).all(), "V140 meta direction invalid")
    signs = np.sign(estimates)
    meta_sign = np.sign(meta)
    sign_agreement = np.mean(signs == meta_sign[None, :], axis=0)
    return meta, {
        "environment_n": environment_n,
        "parameter_n": parameter_n,
        "cochran_q_q10_q50_q90": [float(np.quantile(q, value)) for value in (0.1, 0.5, 0.9)],
        "tau2_q10_q50_q90": [float(np.quantile(tau2, value)) for value in (0.1, 0.5, 0.9)],
        "i2_q10_q50_q90": [float(np.quantile(i2, value)) for value in (0.1, 0.5, 0.9)],
        "stability_q10_q50_q90": [float(np.quantile(stability, value)) for value in (0.1, 0.5, 0.9)],
        "sign_agreement_q10_q50_q90": [float(np.quantile(sign_agreement, value)) for value in (0.1, 0.5, 0.9)],
        "zeroed_parameter_fraction": float(np.mean(stability <= 1e-12)),
        "meta_parameter_sha256": array_sha256(meta),
    }


def fit_meta_parameters(
    state: np.ndarray,
    target: np.ndarray,
    environment: np.ndarray,
    extra_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    estimates: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    supported: list[str] = []
    for name in sorted(pd.unique(environment)):
        mask = environment == name
        class_counts = [int(np.sum(target[mask] == direction)) for direction in (0, 1)]
        if int(mask.sum()) < MIN_ENVIRONMENT_ROWS or min(class_counts) < MIN_ENVIRONMENT_CLASS_ROWS:
            continue
        current_extra = None if extra_weights is None else extra_weights[mask]
        estimate, variance, audit = fit_environment_logit(state[mask], target[mask], current_extra)
        estimates.append(estimate)
        variances.append(variance)
        audits.append({"environment": str(name), **audit})
        supported.append(str(name))
    require(len(estimates) >= 3, "V140 fewer than three supported source-time environments")
    parameters, meta_audit = random_effects_meta(np.vstack(estimates), np.vstack(variances))
    return parameters, {
        "supported_environment_n": len(supported),
        "supported_environments": supported,
        "supported_source_family_n": len({name.split("|")[1] for name in supported}),
        "covered_event_group_n": int(np.sum(np.isin(environment, supported))),
        "covered_event_group_fraction": float(np.mean(np.isin(environment, supported))),
        "environment_fits": audits,
        "meta": meta_audit,
        "coefficient_l2": float(np.linalg.norm(parameters[:STATE_DIMENSION])),
        "intercept": float(parameters[-1]),
    }


def native_meta_bootstrap(
    state: np.ndarray,
    target: np.ndarray,
    environment: np.ndarray,
    target_state: np.ndarray,
    nominal_probability: np.ndarray,
    nominal_parameters: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    probabilities = np.empty((len(target_state), draws), float)
    cosines = np.empty(draws, float)
    nominal_norm = max(float(np.linalg.norm(nominal_parameters[:STATE_DIMENSION])), 1e-12)
    for draw in range(draws):
        event_weight = generator.exponential(size=len(target))
        parameters, _ = fit_meta_parameters(state, target, environment, event_weight)
        probabilities[:, draw] = expit(np.clip(target_state @ parameters[:STATE_DIMENSION] + parameters[-1], -SCORE_CLIP, SCORE_CLIP))
        current = parameters[:STATE_DIMENSION]
        current_norm = max(float(np.linalg.norm(current)), 1e-12)
        cosines[draw] = float(current @ nominal_parameters[:STATE_DIMENSION] / (current_norm * nominal_norm))
    median_probability = np.median(probabilities, axis=1)
    row_std = probabilities.std(axis=1)
    return {
        "contract": "event-group Bayesian complete refit of every supported source-time environment head and random-effects meta direction",
        "draws": draws,
        "seed": seed,
        "event_group_unit_weights": True,
        "environment_heads_completely_refit": True,
        "target_labels_used": False,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(probabilities - nominal_probability[:, None]))),
        "median_probability_mean_absolute_deviation": float(np.mean(np.abs(median_probability - nominal_probability))),
        "direction_agreement_with_nominal": float(np.mean((median_probability >= 0.5) == (nominal_probability >= 0.5))),
        "coefficient_cosine_q10_q50_q90": [float(np.quantile(cosines, q)) for q in (0.1, 0.5, 0.9)],
        "row_probability_std_q10_q50_q90": [float(np.quantile(row_std, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(median_probability),
    }


def meta_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
    native_draws: int = 0,
    native_seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V140 expects one market")
    require(str(ordered.market.iloc[0]) == str(target_frame.market.iloc[0]), "V140 market mismatch")
    processor = numeric.RobustNumericState().fit(ordered)
    train_rows, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_state, train_y, group_ids, environment, environment_audit = group_state_and_environment(ordered, train_rows)
    parameters, fit_audit = fit_meta_parameters(train_state, train_y, environment)
    score = np.clip(target_state @ parameters[:STATE_DIMENSION] + parameters[-1], -SCORE_CLIP, SCORE_CLIP)
    probability = np.clip(expit(score), 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V140 probability invalid")
    native = None if native_draws <= 0 else native_meta_bootstrap(
        train_state, train_y, environment, target_state, probability, parameters, native_draws, native_seed,
    )
    return probability, {
        "market": str(ordered.market.iloc[0]),
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "train_event_group_n": len(group_ids),
        "causal_numeric_and_missing_only": True,
        "numeric_feature_n": len(FEATURES),
        "missing_flag_n": len(FEATURES),
        "source_used_only_for_training_environment": True,
        "source_market_time_target_routing": False,
        "categorical_text_ticker_issuer_return_v69_probability_or_confidence_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "environment_audit": environment_audit,
        "static_spec": STATIC_SPEC,
        "fit_audit": fit_audit,
        "environment_specific_balanced_logit_heads": True,
        "diagonal_observed_fisher_uncertainty": True,
        "der_simonian_laird_random_effects": True,
        "cochran_q_continuous_heterogeneity_shrink": True,
        "pair_list_auc_group_dro_or_worst_class_cvar": False,
        "v69_offset_residual_or_confidence_surface": False,
        "spatial_median_tyler_scatter_or_class_density": False,
        "target_rows_used_for_scale_fit_environment_or_selection": False,
        "target_labels_used": False,
        "threshold_calibration_regularization_or_environment_grid_tuned": False,
        "native_meta_bootstrap": native,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "predicted_up_rate": float(np.mean(probability >= 0.5)),
            "score_q10_q50_q90": [float(np.quantile(score, q)) for q in (0.1, 0.5, 0.9)],
            "probability_sha256": array_sha256(probability),
        },
    }


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger, model_audit = meta_direction(inner_train, inner_valid)
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
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 10.0 * baseline_metric["all_trade_mean_signed_net"],
    }]
    fit = model_audit["fit_audit"]
    model_eligible = bool(
        model_audit["causal_numeric_and_missing_only"]
        and model_audit["source_used_only_for_training_environment"]
        and not model_audit["source_market_time_target_routing"]
        and model_audit["environment_specific_balanced_logit_heads"]
        and model_audit["diagonal_observed_fisher_uncertainty"]
        and model_audit["der_simonian_laird_random_effects"]
        and model_audit["cochran_q_continuous_heterogeneity_shrink"]
        and fit["supported_environment_n"] >= 3
        and fit["covered_event_group_fraction"] >= 0.80
        and all(item["optimizer_success"] for item in fit["environment_fits"])
        and not model_audit["pair_list_auc_group_dro_or_worst_class_cvar"]
        and not model_audit["v69_offset_residual_or_confidence_surface"]
        and not model_audit["target_rows_used_for_scale_fit_environment_or_selection"]
        and not model_audit["target_labels_used"]
        and not model_audit["threshold_calibration_regularization_or_environment_grid_tuned"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": architecture["name"],
            "architecture": architecture,
            "metrics": current,
            "auc_delta": auc_delta,
            "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"] + 10.0 * current["all_trade_mean_signed_net"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 or fixed .25/.50 source-time random-effects meta-logit blend; fixed 2*AUC+BA+10*net with BA/net guards",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "environment_definition_meta_method_l2_or_blend_micro_tuning": False,
    }, model_audit


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v140_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in (fold_specs[:1] if smoke else fold_specs):
        outer_champion = champion.loc[
            champion.market.eq(market) & champion.fold.eq(fold)
        ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V140 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
        challenger, outer_model_audit = meta_direction(
            outer_train,
            outer_valid,
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
        diagnostic.loc[positions, "v140_model"] = selected["name"]
        outer_diagnostics = {
            architecture["name"]: metric(
                outer_valid,
                blend_probability(baseline, challenger, architecture["weight"]),
                confidence,
                high,
            ) for architecture in ARCHITECTURES
        }
        baseline_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["meta_logit_prob"] = challenger
        evidence["candidate_prob"] = candidate
        evidence["v140_model"] = selected["name"]
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
            "outer_baseline": baseline_metric,
            "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V140 RANDOM EFFECTS META] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V140 evidence repeats event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V140 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V140 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(
        work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"),
        timestamp.dt.strftime("KR|%Y-%m-%d"),
    )
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V140 bootstrap blocks")
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
            balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)
        ))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(
            np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns)
            - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)
        ))

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
        "standardized": True,
        "method": "paired market-fold-time block bootstrap over inner-locked V140 source-time random-effects meta policy",
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
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V140 canonical mismatch")
    baseline = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = bool(
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    meta_contract = all(
        audit[side]["causal_numeric_and_missing_only"]
        and audit[side]["source_used_only_for_training_environment"]
        and not audit[side]["source_market_time_target_routing"]
        and audit[side]["environment_specific_balanced_logit_heads"]
        and audit[side]["diagonal_observed_fisher_uncertainty"]
        and audit[side]["der_simonian_laird_random_effects"]
        and audit[side]["cochran_q_continuous_heterogeneity_shrink"]
        and audit[side]["fit_audit"]["supported_environment_n"] >= 3
        and audit[side]["fit_audit"]["covered_event_group_fraction"] >= 0.80
        and not audit[side]["pair_list_auc_group_dro_or_worst_class_cvar"]
        and not audit[side]["v69_offset_residual_or_confidence_surface"]
        and not audit[side]["target_rows_used_for_scale_fit_environment_or_selection"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["threshold_calibration_regularization_or_environment_grid_tuned"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
    )
    model_native = [audit["outer_model_audit"]["native_meta_bootstrap"] for audit in audits]
    model_native_valid = all(
        item is not None
        and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"]
        and item["environment_heads_completely_refit"]
        and not item["target_labels_used"]
        and 0.0 <= item["direction_agreement_with_nominal"] <= 1.0
        for item in model_native
    )
    native = candidate_summary["robustness"]["bootstrap"]
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
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native),
        "meta_native_bootstrap_contract_verified": model_native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "source_market_time_random_effects_meta_contract_verified": meta_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V140 fallback not exact entire V69")
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
        "nested_bootstrap": nested,
        "candidate_native_robustness_bootstrap": native,
        "model_native_meta_bootstrap": model_native,
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
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "model_native_meta_bootstrap": evaluation["model_native_meta_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V140 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V140 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V140 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V140").resolve(), "V140 output is outside controller research/staging/V140")
    require(out.exists() and out.is_dir(), "V140 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V140 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V140 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True,
        "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing),
        "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V140 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V140 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V140",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V140_DIAGNOSTIC_RANDOM_EFFECTS_META_LOGIT": evaluation["candidate_summary"],
                "V140_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact,
            "authority_audit": authority,
            "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V140",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "opened_dev_only": True,
            "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V140",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "environment_meta_audits": [
                {"market": audit["market"], "fold": audit["fold"], "inner": audit["inner_model_audit"]["fit_audit"], "outer": audit["outer_model_audit"]["fit_audit"]}
                for audit in audits
            ],
            "material_gate": gate,
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "V140_RANDOM_EFFECTS_META_LOGIT_REPORT.json": {
            "version": "V140",
            "hypothesis": HYPOTHESIS,
            "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES,
            "nested_fold_audits": audits,
            "evaluation": compact,
            "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V140",
            "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap",
            "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V140",
            "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
            "model_native_meta_bootstrap": evaluation["model_native_meta_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V140",
            "status": "MATCH",
            "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION,
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "material_pass": evaluation["material_pass"],
            "champion_changed": evaluation["material_pass"],
            "seal_state": "UNOPENED",
            "seal_authorized": False,
            "output_conflict_audit": conflict_audit,
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V140_RANDOM_EFFECTS_META_LOGIT_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V140_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID,
        "files": files,
    }, out / "ARTIFACT_MANIFEST.json")
    return {
        "output": str(out),
        "artifact_count": len(files) + 1,
        "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json"),
    }


def support_probe_kr2(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    outer_champion = champion.loc[
        champion.market.eq("KR") & champion.fold.eq(2)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS["KR"], "KR fold 2 size changed")
    outer_valid = aligned_dev(dev, outer_champion)
    start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_train = prior_train(dev, "KR", start, outer_valid)
    outer_chronology = chronology(outer_train, outer_valid, "V140 KR2 support outer")
    inner_train, inner_valid, _, inner_chronology = inner_partition(dev, champion, "KR", start)
    inner_probability, inner_audit = meta_direction(inner_train, inner_valid)
    outer_probability, outer_audit = meta_direction(outer_train, outer_valid)
    require(np.isfinite(inner_probability).all() and np.isfinite(outer_probability).all(), "V140 KR2 support probability invalid")
    return {
        "status": "KR2_SUPPORT_OK_NO_OUTER_EVALUATION",
        "inner_target_n": len(inner_valid),
        "outer_target_n": len(outer_valid),
        "inner_train_event_group_n": inner_audit["train_event_group_n"],
        "outer_train_event_group_n": outer_audit["train_event_group_n"],
        "inner_supported_environment_n": inner_audit["fit_audit"]["supported_environment_n"],
        "outer_supported_environment_n": outer_audit["fit_audit"]["supported_environment_n"],
        "inner_supported_environments": inner_audit["fit_audit"]["supported_environments"],
        "outer_supported_environments": outer_audit["fit_audit"]["supported_environments"],
        "inner_coverage": inner_audit["fit_audit"]["covered_event_group_fraction"],
        "outer_coverage": outer_audit["fit_audit"]["covered_event_group_fraction"],
        "inner_probability_sha256": array_sha256(inner_probability),
        "outer_probability_sha256": array_sha256(outer_probability),
        "inner_chronology": inner_chronology,
        "outer_chronology": outer_chronology,
        "outer_labels_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=not bool(CONTROLLER_OUTPUT))
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--support-probe-kr2", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe_kr2
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V140 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V140 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = args.audit_only or args.smoke_test or args.support_probe_kr2
    runtime = numeric.configure_bounded_runtime() if bounded else {
        "mode": "controller_bound_full",
        "thread_policy": "authorized parent inherited",
        "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK",
            "hypothesis": HYPOTHESIS,
            "authority": authority,
            "runtime": runtime,
            "dev_rows": len(dev),
            "v69_rows": len(champion),
            "features": FEATURES,
            "static_spec": STATIC_SPEC,
            "architecture": "source-time environment logits combined coordinatewise by Fisher-variance random-effects meta-analysis and continuous heterogeneity shrink",
            "causal_numeric_and_missing_only": True,
            "source_used_only_for_training_environment": True,
            "target_rows_or_labels_used": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"],
                "current_material_gate": True,
                "nested_bootstrap_key": "selected.material_gate.nested_bootstrap",
                "standardized_nested_bootstrap": True,
                "controller_native_robustness_bootstrap": True,
                "model_native_meta_bootstrap": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        if args.support_probe_kr2:
            result = support_probe_kr2(dev, champion)
            result["runtime"] = runtime
            result["gpu_model_calls"] = 0
            result["output_written"] = False
            print(json.dumps(clean(result), ensure_ascii=False, indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
        evaluation = evaluate(
            champion,
            diagnostic,
            evidence,
            audits,
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
        summary.update({
            "raw_inner_selected": audits[0]["policy"]["selected"],
            "raw_inner_best_non_noop": best_non_noop,
            "raw_outer_baseline": audits[0]["outer_baseline"],
            "raw_outer_selected": audits[0]["outer_candidate"],
            "raw_outer_best_non_noop_evaluation_only": audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]],
            "inner_model_audit": audits[0]["inner_model_audit"],
            "outer_model_audit": audits[0]["outer_model_audit"],
            "candidate_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        })
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
