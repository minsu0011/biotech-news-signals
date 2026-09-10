"""V145 source-time quantile-transport normalized direction challenger.

Every strict-past same-market cut collapses immutable V36 causal state to equal
event-group rows.  Three label-independent chronological past terciles combine
with source family to define nuisance environments.  Each observed numeric
value is mapped through its supported environment empirical marginal CDF and
then through the full-past pooled marginal inverse CDF.  Unsupported source-time
cells use the matching time-tercile reference, then the pooled reference.  A
future query uses only its source's latest supported past-tercile transport.
Missingness is preserved explicitly and no target batch distribution is read.

One shared class-balanced fixed ridge-logistic head learns direction on the
transported 37 numeric fields plus 37 missing flags.  Unlike V143, no nuisance
SVD/subspace is estimated; unlike V140, there are no environment heads or
random-effects meta-analysis; unlike V86, the transform is environment-specific
pooled-scale transport rather than one global Gaussian copula followed by QDA;
unlike V85, no past row is reweighted.  Only exact V69 noop versus the single
full challenger is selected inner-only.  Outer labels are evaluation-only under
a strict 35-minute embargo/event purge; V69 confidence/high_conf stay exact and
any material failure restores the exact entire atomic V69 frame.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import experiment_v142_chronological_delete_block_sign_intersection as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V145_SOURCE_TIME_QUANTILE_TRANSPORT_NORMALIZED_V1"
VERSION = 145
HYPOTHESIS = "SOURCE_TIME_QUANTILE_TRANSPORT_NORMALIZED_DIRECTION_V1"
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
EXPECTED_V69_EXPERIMENT_ID = scaffold.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v142_chronological_delete_block_sign_intersection.py"
EXPECTED_SCAFFOLD_SHA256 = "96e72094387e7da1fd45f7249e9b3196d31a6ced9666ddde1054c126149fdc4c"

FEATURES = scaffold.FEATURES
EMBARGO = scaffold.EMBARGO
INNER_VALID_FRACTION = scaffold.INNER_VALID_FRACTION
TIME_TERCILES = 3
MIN_SOURCE_TIME_ROWS = 30
MIN_FEATURE_OBSERVED = 12
LOGISTIC_C = 0.5
PROBABILITY_EPSILON = 1e-5
STATE_CLIP = 8.0
SEED = 14501
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
NATIVE_BOOTSTRAP_DRAWS = 128
SMOKE_NATIVE_BOOTSTRAP_DRAWS = 32
SMOKE_THREADS = 2
PREPARATION_CPU_AFFINITY = (30, 31)
ARCHITECTURES = ({"name": "SOURCE_TIME_QUANTILE_TRANSPORT_FULL"},)

require = scaffold.require
sha256 = scaffold.sha256
array_sha256 = scaffold.array_sha256
clean = scaffold.clean
bool_series = scaffold.bool_series
atomic_json = scaffold.atomic_json
atomic_csv = scaffold.atomic_csv
load_authorized = scaffold.load_authorized
aligned_dev = scaffold.aligned_dev
chronology = scaffold.chronology
metric = scaffold.metric
controller = scaffold.controller
v44 = scaffold.v44


def verify_static_spec() -> dict[str, Any]:
    payload = (
        "equal-event-group|37-causal-numeric-plus-37-missing|source-family-x-three-chronological-terciles|"
        "environment-empirical-marginal-cdf|full-past-pooled-marginal-inverse-cdf|"
        "source-time-min30-time-tercile-then-pooled-fallback|future-latest-supported-past-env|"
        "no-target-batch|one-shared-class-balanced-ridge-logit-C0.5|noop-versus-full"
    )
    require(len(FEATURES) == 37 and TIME_TERCILES == 3 and MIN_SOURCE_TIME_ROWS == 30 and LOGISTIC_C == 0.5, "V145 static spec changed")
    require(len(ARCHITECTURES) == 1, "V145 architecture grid forbidden")
    return {
        "causal_numeric_n": len(FEATURES), "missing_flag_n": len(FEATURES),
        "state_dimension": 2 * len(FEATURES), "time_terciles": TIME_TERCILES,
        "minimum_source_time_rows": MIN_SOURCE_TIME_ROWS, "minimum_feature_observed": MIN_FEATURE_OBSERVED,
        "transport": "environment empirical marginal CDF to full-past pooled marginal inverse CDF",
        "fallback": "same chronological tercile across sources, then pooled marginal",
        "target_environment": "target source family latest supported strict-past tercile, else latest time-tercile fallback",
        "target_batch_statistics": False,
        "head": "one shared class-balanced L2 logistic C=0.5 liblinear with intercept",
        "architecture_candidates_excluding_noop": 1,
        "tercile_support_C_fallback_or_policy_grid": False,
        "representation_spec_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V142 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name, "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, metrics, canonical gate, bootstrap and atomic IO only; no V142 output is read",
    }
    audit["v145_access_contract"] = {
        "immutable_v36_dev_only": True, "atomic_output_v69_only": True,
        "failed_outputs_read": False, "dev_extension_read": False,
        "role_assignment_read": False, "research_seal_read": False,
        "final_reserve_read": False, "prior_version_output_read": False,
        "target_batch_distribution_read": False,
    }
    return audit


def preparation_affinity() -> dict[str, Any]:
    audit = scaffold.preparation_affinity()
    require(tuple(audit["actual_logical_cpus"]) == PREPARATION_CPU_AFFINITY, "V145 bounded affinity changed")
    return audit


def strict_reference(dev: pd.DataFrame, champion: pd.DataFrame, market: str, target_valid: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_start = pd.Timestamp(target_valid.event_time_utc.min())
    reference_champion = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < target_start - EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(reference_champion) >= 120, f"V145 {label} reference too small")
    reference = aligned_dev(dev, reference_champion)
    audit = chronology(reference, target_valid, f"V145 {label} source-time quantile transport")
    audit["reference_source"] = "strict-past same-market atomic V69 OOF IDs joined to immutable V36 causal state and source_family"
    audit["target_labels_used_for_fit"] = False
    audit["target_batch_statistics_used"] = False
    return reference, audit


def inner_partition(dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[
        champion.market.eq(market) & (champion.event_time_utc < outer_start - EMBARGO)
    ].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(prior) >= (1000 if market == "US" else 220), f"insufficient prior {market} V69 OOF")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_reference, audit = strict_reference(dev, champion, market, inner_valid, f"inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_valid, inner_champion, inner_reference, audit


def source_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.source_family.fillna("__MISSING_SOURCE__").astype(str).to_numpy()


def collapse_event_groups(reference: pd.DataFrame) -> pd.DataFrame:
    columns = ["event_group_id", "event_time_utc", "market", "source_family", "y", *FEATURES]
    work = reference[columns].copy()
    work["source_family"] = work.source_family.fillna("__MISSING_SOURCE__").astype(str)
    grouped = work.groupby("event_group_id", sort=True)
    require(int(grouped.y.nunique().max()) == 1, "V145 event group crosses labels")
    require(int(grouped.source_family.nunique().max()) == 1, "V145 event group crosses source families")
    result = grouped.agg({
        "event_time_utc": "min", "market": "first", "source_family": "first", "y": "first",
        **{feature: "mean" for feature in FEATURES},
    }).reset_index().sort_values(["event_time_utc", "event_group_id"], kind="stable").reset_index(drop=True)
    require(result.event_group_id.is_unique, "V145 event group collapse invalid")
    return result


def class_balanced_weights(target: np.ndarray, base_weight: np.ndarray | None = None) -> np.ndarray:
    target = np.asarray(target, int)
    weight = np.ones(len(target), float) if base_weight is None else np.asarray(base_weight, float).copy()
    require(weight.shape == target.shape and np.all(weight > 0) and np.isfinite(weight).all(), "V145 weights invalid")
    masses = np.asarray([weight[target == label].sum() for label in (0, 1)], float)
    require(np.all(masses > 0), "V145 class support missing")
    total = float(weight.sum())
    weight *= np.where(target == 1, total / (2.0 * masses[1]), total / (2.0 * masses[0]))
    return weight / float(np.mean(weight))


def chronological_tercile(rows: int) -> np.ndarray:
    require(rows >= TIME_TERCILES * 40, "V145 chronological support too small")
    result = np.minimum((np.arange(rows) * TIME_TERCILES) // rows, TIME_TERCILES - 1).astype(int)
    require(np.all(np.bincount(result, minlength=TIME_TERCILES) >= 40), "V145 chronological tercile support changed")
    return result


def mid_cdf(reference: np.ndarray, value: float) -> float:
    if not math.isfinite(value) or len(reference) == 0:
        return 0.5
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    return float((left + right + 1.0) / (2.0 * (len(reference) + 1.0)))


def inverse_sorted_quantile(reference: np.ndarray, quantile: float) -> float:
    require(len(reference) > 0, "V145 inverse pooled reference empty")
    position = float(np.clip(quantile, 0.0, 1.0)) * (len(reference) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(reference) - 1)
    fraction = position - lower
    return float((1.0 - fraction) * reference[lower] + fraction * reference[upper])


class SourceTimeQuantileTransport:
    def __init__(self) -> None:
        self.pooled: list[np.ndarray] = []
        self.time_refs: dict[tuple[int, int], np.ndarray] = {}
        self.source_time_refs: dict[tuple[str, int, int], np.ndarray] = {}
        self.supported_source_time: set[tuple[str, int]] = set()
        self.median = np.empty(0, float)
        self.scale = np.empty(0, float)
        self.training_tercile = np.empty(0, int)
        self.fit_audit: dict[str, Any] = {}

    def fit(self, groups: pd.DataFrame) -> "SourceTimeQuantileTransport":
        raw = groups.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        raw[~np.isfinite(raw)] = np.nan
        source = source_values(groups)
        tercile = chronological_tercile(len(groups))
        self.training_tercile = tercile
        self.pooled = []
        for feature_index in range(len(FEATURES)):
            values = np.sort(raw[np.isfinite(raw[:, feature_index]), feature_index])
            require(len(values) >= MIN_FEATURE_OBSERVED, "V145 pooled feature support missing")
            self.pooled.append(values)
        for time_id in range(TIME_TERCILES):
            mask = tercile == time_id
            for feature_index in range(len(FEATURES)):
                values = np.sort(raw[mask & np.isfinite(raw[:, feature_index]), feature_index])
                self.time_refs[(time_id, feature_index)] = values if len(values) >= MIN_FEATURE_OBSERVED else self.pooled[feature_index]
        cell_rows: list[dict[str, Any]] = []
        for family in sorted(set(source)):
            for time_id in range(TIME_TERCILES):
                mask = (source == family) & (tercile == time_id)
                supported = int(mask.sum()) >= MIN_SOURCE_TIME_ROWS
                if supported:
                    self.supported_source_time.add((str(family), time_id))
                feature_support = 0
                for feature_index in range(len(FEATURES)):
                    values = np.sort(raw[mask & np.isfinite(raw[:, feature_index]), feature_index])
                    if supported and len(values) >= MIN_FEATURE_OBSERVED:
                        self.source_time_refs[(str(family), time_id, feature_index)] = values
                        feature_support += 1
                cell_rows.append({"source_family": str(family), "time_tercile": time_id, "rows": int(mask.sum()), "supported": supported, "supported_features": feature_support})
        transported, transform_audit = self._transform(raw, source, tercile, training=True)
        self.median = np.nanmedian(transported, axis=0)
        self.median[~np.isfinite(self.median)] = 0.0
        lower, upper = np.nanquantile(transported, [0.25, 0.75], axis=0)
        spread = upper - lower
        fallback = np.nanstd(transported, axis=0)
        self.scale = np.where(np.isfinite(spread) & (spread > 1e-9), spread / 1.349, np.where(np.isfinite(fallback) & (fallback > 1e-9), fallback, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V145 transported robust scale invalid")
        self.fit_audit = {
            "rows": len(groups), "source_family_n": int(len(set(source))),
            "time_tercile_counts": np.bincount(tercile, minlength=TIME_TERCILES).astype(int).tolist(),
            "source_time_cell_n": len(cell_rows), "supported_source_time_cell_n": len(self.supported_source_time),
            "cell_inventory": cell_rows, "training_transform": transform_audit,
            "pooled_reference_sha256": hashlib.sha256("\n".join(array_sha256(values) for values in self.pooled).encode("utf-8")).hexdigest(),
        }
        return self

    def _reference(self, family: str, time_id: int, feature_index: int) -> tuple[np.ndarray, str]:
        key = (family, time_id, feature_index)
        if key in self.source_time_refs:
            return self.source_time_refs[key], "SOURCE_TIME"
        time_key = (time_id, feature_index)
        if time_key in self.time_refs:
            return self.time_refs[time_key], "TIME"
        return self.pooled[feature_index], "POOLED"

    def _transform(self, raw: np.ndarray, source: np.ndarray, tercile: np.ndarray, training: bool) -> tuple[np.ndarray, dict[str, Any]]:
        result = np.full(raw.shape, np.nan, float)
        levels = {"SOURCE_TIME": 0, "TIME": 0, "POOLED": 0}
        observed = np.isfinite(raw)
        for row in range(len(raw)):
            family, time_id = str(source[row]), int(tercile[row])
            for feature_index in range(len(FEATURES)):
                if not observed[row, feature_index]:
                    continue
                reference, level = self._reference(family, time_id, feature_index)
                u = np.clip(mid_cdf(reference, float(raw[row, feature_index])), 1e-4, 1.0 - 1e-4)
                result[row, feature_index] = inverse_sorted_quantile(self.pooled[feature_index], u)
                levels[level] += 1
        return result, {
            "rows": len(raw), "training_rows": training, "target_batch_statistics_used": False,
            "observed_values": int(observed.sum()), "missing_values": int((~observed).sum()),
            "reference_level_counts": levels,
        }

    def transform_training(self, groups: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        raw = groups.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        raw[~np.isfinite(raw)] = np.nan
        transported, audit = self._transform(raw, source_values(groups), self.training_tercile, training=True)
        observed = np.isfinite(transported)
        filled = np.where(observed, transported, self.median)
        numeric = np.clip((filled - self.median) / self.scale, -STATE_CLIP, STATE_CLIP)
        state = np.column_stack([numeric, (~observed).astype(float)])
        require(state.shape == (len(groups), 2 * len(FEATURES)) and np.isfinite(state).all(), "V145 training transport invalid")
        audit.update({"state_sha256": array_sha256(state), "missing_rate": float((~observed).mean())})
        return state, audit

    def transform_target(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        raw = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        raw[~np.isfinite(raw)] = np.nan
        latest = np.full(len(frame), TIME_TERCILES - 1, int)
        transported, audit = self._transform(raw, source_values(frame), latest, training=False)
        observed = np.isfinite(transported)
        filled = np.where(observed, transported, self.median)
        numeric = np.clip((filled - self.median) / self.scale, -STATE_CLIP, STATE_CLIP)
        state = np.column_stack([numeric, (~observed).astype(float)])
        require(state.shape == (len(frame), 2 * len(FEATURES)) and np.isfinite(state).all(), "V145 target transport invalid")
        audit.update({"state_sha256": array_sha256(state), "missing_rate": float((~observed).mean()), "target_time_assignment": "latest strict-past tercile only"})
        return state, audit


def fit_head(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[LogisticRegression, dict[str, Any]]:
    model = LogisticRegression(
        penalty="l2", C=LOGISTIC_C, fit_intercept=True, solver="liblinear",
        tol=1e-8, max_iter=500, random_state=SEED,
    )
    model.fit(design, target, sample_weight=weight)
    coefficient = np.asarray(model.coef_[0], float)
    intercept = float(model.intercept_[0])
    require(np.isfinite(coefficient).all() and math.isfinite(intercept) and int(model.n_iter_[0]) < 500, "V145 shared ridge head failed")
    return model, {
        "iterations": int(model.n_iter_[0]), "coefficient_l2": float(np.linalg.norm(coefficient)),
        "intercept": intercept, "coefficient_sha256": array_sha256(coefficient),
        "head_sha256": hashlib.sha256((array_sha256(coefficient) + f"|{intercept:.17g}").encode("utf-8")).hexdigest(),
    }


def native_head_bootstrap(design: np.ndarray, target: np.ndarray, target_design: np.ndarray, nominal: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    samples = np.empty((len(target_design), draws), float)
    hashes: list[str] = []
    for draw in range(draws):
        weight = class_balanced_weights(target, generator.exponential(1.0, len(target)))
        model, audit = fit_head(design, target, weight)
        samples[:, draw] = np.clip(model.predict_proba(target_design)[:, 1], PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        hashes.append(audit["head_sha256"])
    std = np.std(samples, axis=1)
    return {
        "contract": "equal-event-group Bayesian bootstrap complete refit of shared ridge head with label-free quantile transport frozen",
        "draws": draws, "seed": seed, "event_group_unit_weights": True,
        "shared_head_complete_refit": True, "transport_refit": False,
        "transport_is_label_free": True, "target_labels_used": False,
        "probability_mean_absolute_deviation": float(np.mean(np.abs(samples - nominal[:, None]))),
        "row_probability_std_q10_q50_q90": [float(np.quantile(std, q)) for q in (0.1, 0.5, 0.9)],
        "median_probability_sha256": array_sha256(np.median(samples, axis=1)),
        "head_hash_inventory_sha256": hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest(),
    }


def transport_direction(reference: pd.DataFrame, target_frame: pd.DataFrame, native_draws: int = 0, native_seed: int = SEED) -> tuple[np.ndarray, dict[str, Any]]:
    require(reference.market.nunique() == target_frame.market.nunique() == 1 and str(reference.market.iloc[0]) == str(target_frame.market.iloc[0]), "V145 market mismatch")
    groups = collapse_event_groups(reference)
    transporter = SourceTimeQuantileTransport().fit(groups)
    design, train_transport = transporter.transform_training(groups)
    target_design, target_transport = transporter.transform_target(target_frame)
    target = groups.y.to_numpy(int)
    model, head_audit = fit_head(design, target, class_balanced_weights(target))
    probability = np.clip(model.predict_proba(target_design)[:, 1], PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    native = None if native_draws <= 0 else native_head_bootstrap(design, target, target_design, probability, native_draws, native_seed)
    return probability, {
        "market": str(reference.market.iloc[0]), "reference_row_n": len(reference),
        "reference_event_group_n": len(groups), "target_n": len(target_frame),
        "causal_pre_event_only": True, "strict_past_atomic_v69_oof_ids_only": True,
        "equal_event_group_training": True, "causal_numeric_feature_n": len(FEATURES),
        "missing_flag_n": len(FEATURES), "source_feature_role": "label-free transport environment only",
        "text_ticker_issuer_return_v69_probability_confidence_or_external_feature_n": 0,
        "architecture": "source-family by chronological-tercile empirical marginal CDF to pooled inverse marginal transport plus one shared ridge-logistic head",
        "static_spec": STATIC_SPEC, "transport_fit_audit": transporter.fit_audit,
        "train_transport": train_transport, "target_transport": target_transport,
        "head_audit": head_audit, "native_transport_head_bootstrap": native,
        "target_rows_used_for_transport_fit_scale_or_head": False, "target_labels_used": False,
        "target_batch_statistics_used": False, "source_routed_prediction_head": False,
        "environment_head_random_effects_or_nuisance_subspace": False,
        "global_gaussian_copula_or_qda": False, "entropy_or_importance_weighting": False,
        "tercile_support_C_fallback_or_policy_grid": False,
        "probability_sha256": array_sha256(probability),
    }


def choose_inner(inner_valid: pd.DataFrame, inner_champion: pd.DataFrame, inner_reference: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = inner_champion.prob.to_numpy(float)
    candidate, model_audit = transport_direction(inner_reference, inner_valid)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    base_metric = metric(inner_valid, baseline, confidence, high)
    candidate_metric = metric(inner_valid, candidate, confidence, high)
    auc_delta = candidate_metric["auc"] - base_metric["auc"]
    ba_delta = candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"]
    net_delta = candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"]
    trials = [{
        "name": "V69_NOOP", "architecture": None, "metrics": base_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "eligible": True, "score": 2.0 * base_metric["auc"] + base_metric["balanced_accuracy"],
    }, {
        "name": ARCHITECTURES[0]["name"], "architecture": ARCHITECTURES[0], "metrics": candidate_metric,
        "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta, "all_trade_net_delta": net_delta,
        "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
        "score": 2.0 * candidate_metric["auc"] + candidate_metric["balanced_accuracy"],
    }]
    selected = max((trial for trial in trials if trial["eligible"]), key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"))
    return {
        "selection_rule": "inner-past only; exact V69 noop versus one fixed source-time quantile-transport shared head; maximize 2*AUC+BA with fixed BA/net safety guard",
        "baseline": base_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "tercile_support_C_fallback_transport_or_policy_tuned": False,
    }, model_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v145_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    folds = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in ([('US', 2)] if smoke else folds):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, f"outer {market} fold {fold}")
        inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_model_audit = choose_inner(inner_valid, inner_champion, inner_reference)
        full_probability, outer_model_audit = transport_direction(
            outer_reference, outer_valid,
            native_draws=SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS,
            native_seed=SEED + 100 * fold + (0 if market == "US" else 1000),
        )
        baseline = outer_champion.prob.to_numpy(float)
        candidate = baseline.copy() if policy["selected"]["architecture"] is None else full_probability
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        base_metric = metric(outer_valid, baseline, confidence, high)
        full_metric = metric(outer_valid, full_probability, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        lookup = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(lookup)
        diagnostic.loc[positions, "v145_model"] = policy["selected"]["name"]
        evidence = outer_champion[["event_id", "event_group_id", "event_time_utc", "market", "ticker", "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf"]].copy()
        evidence["baseline_prob"] = baseline
        evidence["full_transport_prob"] = full_probability
        evidence["candidate_prob"] = candidate
        evidence["v145_model"] = policy["selected"]["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold, "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_model_audit": inner_model_audit, "outer_model_audit": outer_model_audit,
            "outer_full_architecture_evaluation_only": full_metric,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True, "outer_labels_used_for_selection": False,
        })
        print(f"[V145 QUANTILE TRANSPORT] market={market} fold={fold} selected={policy['selected']['name']} inner_auc_delta={policy['trials'][1]['auc_delta']:+.6f} outer_full_auc_delta={full_metric['auc'] - base_metric['auc']:+.6f}", flush=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V145 evidence repeats event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V145 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V145 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V145 bootstrap blocks")
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
        values["balanced_accuracy_delta"].append(float(balanced_accuracy_score(target, candidate >= 0.5) - balanced_accuracy_score(target, baseline >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        values["all_trade_net_delta"].append(float(np.mean(np.where(candidate >= 0.5, 1.0, -1.0) * returns) - np.mean(np.where(baseline >= 0.5, 1.0, -1.0) * returns)))
    def interval(numbers: list[float]) -> dict[str, Any]:
        array = np.asarray(numbers, float)
        return {"effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)), "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)), "probability_gt_zero": float(np.mean(array > 0.0))}
    return {
        "contract_key": "selected.material_gate.nested_bootstrap", "standardized": True,
        "method": "paired market-fold-time block bootstrap over inner-locked V145 source-time quantile-transport policy",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        **{key: interval(numbers) for key, numbers in values.items()},
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int, smoke: bool) -> dict[str, Any]:
    baseline_summary = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "V145 canonical gate count changed")
    require(baseline_summary["research_gate"] == canonical_baseline and candidate_summary["research_gate"] == canonical_candidate, "V145 canonical mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, bool_series(evidence.high_conf))
    nested = paired_nested_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics, candidate_metrics = baseline_summary["metrics"], candidate_summary["metrics"]
    confidence_exact = bool(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool)))
    full_models = [audit["outer_model_audit"] for audit in audits]
    model_valid = all(
        item["causal_pre_event_only"] and item["strict_past_atomic_v69_oof_ids_only"] and item["equal_event_group_training"]
        and not item["target_rows_used_for_transport_fit_scale_or_head"] and not item["target_labels_used"]
        and not item["target_batch_statistics_used"] and not item["source_routed_prediction_head"]
        and not item["environment_head_random_effects_or_nuisance_subspace"] and not item["global_gaussian_copula_or_qda"]
        and not item["entropy_or_importance_weighting"] and not item["tercile_support_C_fallback_or_policy_grid"]
        for item in full_models
    )
    native = [item["native_transport_head_bootstrap"] for item in full_models]
    native_valid = all(
        item is not None and item["draws"] >= (SMOKE_NATIVE_BOOTSTRAP_DRAWS if smoke else NATIVE_BOOTSTRAP_DRAWS)
        and item["event_group_unit_weights"] and item["shared_head_complete_refit"]
        and item["transport_is_label_free"] and not item["target_labels_used"]
        for item in native
    )
    controller_native = candidate_summary["robustness"]["bootstrap"]
    required_native = {"balanced_accuracy_lower95", "balanced_accuracy_upper95", "highconf_strategy_net_lower95", "highconf_strategy_net_upper95"}
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "controller_native_robustness_bootstrap_keys": required_native.issubset(controller_native),
        "model_native_transport_head_bootstrap_verified": native_valid,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "source_time_quantile_transport_contract_verified": model_valid,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    if not material_pass:
        require(selected_frame.equals(champion), "V145 fallback not exact V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"], "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "nested_bootstrap": nested, "controller_native_robustness_bootstrap": controller_native,
        "model_native_transport_head_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "fallback": {"activated": not material_pass, "policy": "exact entire V69 DataFrame" if not material_pass else None, "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def current_material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "contract": HYPOTHESIS, "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())), "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"], "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["controller_native_robustness_bootstrap"],
        "model_native_transport_head_bootstrap": evaluation["model_native_transport_head_bootstrap"],
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V145 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V145 output not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V145").resolve(), "V145 output outside controller staging")
    require(out.exists() and out.is_dir(), "V145 controller stage missing")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing) and not (existing - allowed), "V145 controller prefile conflict")
    return {"controller_bound": True, "required_controller_prefiles": sorted(required), "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": []}


def write_outputs(out: Path, authority: dict[str, Any], evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    conflict = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = current_material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    reports = {
        "MODEL_COMPARISON.json": {"version": "V145", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V145_DIAGNOSTIC_QUANTILE_TRANSPORT": evaluation["candidate_summary"], "V145_FAIL_CLOSED_SELECTED": selected}, "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits},
        "DEV_ROBUSTNESS_REPORT.json": {"version": "V145", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"], "material_gate": gate, "material_checks": evaluation["material_checks"], "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False},
        "SOURCE_TRANSFER_REPORT.json": {"version": "V145", "hypothesis": HYPOTHESIS, "status": evaluation["status"], "selected_by_source_family": selected["metrics"]["by_source_family"], "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"], "selected_by_market": selected["metrics"]["by_market"], "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"], "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"], "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"]},
        "V145_SOURCE_TIME_QUANTILE_TRANSPORT_REPORT.json": {"version": "V145", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC, "architectures": ARCHITECTURES, "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority},
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {"version": "V145", "status": "MATCH", "canonical_check_count": 14, "reported": selected["research_gate"], "controller_recomputed": evaluation["canonical_gate_audit"]["selected"], "material_gate": gate},
        "RUN_STATUS.json": {"version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"], "material_pass": evaluation["material_pass"], "output_conflict_audit": conflict, "seal_state": "UNOPENED", "seal_authorized": False},
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V145_QUANTILE_TRANSPORT_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V145_DIAGNOSTIC_QUANTILE_TRANSPORT_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V145_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def support_probe(dev: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    market, fold = "KR", 2
    outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy().sort_values(["event_time_utc", "event_id"], kind="stable")
    outer_valid = aligned_dev(dev, outer_champion)
    outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
    outer_reference, outer_chronology = strict_reference(dev, champion, market, outer_valid, "KR:2 support outer")
    inner_valid, inner_champion, inner_reference, inner_chronology = inner_partition(dev, champion, market, outer_start)
    policy, inner_audit = choose_inner(inner_valid, inner_champion, inner_reference)
    inner_probability, _ = transport_direction(inner_reference, inner_valid)
    outer_probability, outer_audit = transport_direction(outer_reference, outer_valid)
    return {
        "status": "KR2_SUPPORT_OK", "market": market, "fold": fold, "policy": policy,
        "inner_baseline": metric(inner_valid, inner_champion.prob, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "inner_full_transport": metric(inner_valid, inner_probability, inner_champion.confidence_signal, bool_series(inner_champion.high_conf)),
        "outer_baseline": metric(outer_valid, outer_champion.prob, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "outer_full_transport_evaluation_only": metric(outer_valid, outer_probability, outer_champion.confidence_signal, bool_series(outer_champion.high_conf)),
        "inner_model_audit": inner_audit, "outer_model_audit": outer_audit,
        "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
        "outer_labels_used_for_fit_or_selection": False, "output_written": False,
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
        require(not args.audit_only and not args.smoke_test and not args.support_probe_kr2 and not args.full_run and args.output is None, "explicit mode/output conflicts with controller-bound no-argument V145 full run")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V145 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = args.audit_only or args.smoke_test or args.support_probe_kr2
    affinity = preparation_affinity() if bounded else {"reason": "controller-authorized full run is not bounded preparation"}
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority,
            "dev_rows": len(dev), "v69_rows": len(champion), "features": FEATURES, "static_spec": STATIC_SPEC,
            "architecture": "source-family by chronological-tercile empirical marginal CDF to pooled inverse-marginal quantile transport plus one shared ridge-logistic direction head",
            "causal_pre_event_only": True, "target_batch_statistics_used": False,
            "controller_no_arg_contract": {"environment_variable": "MARKET_BIO_VERSION_OUTPUT", "implicit_mode": "full_run", "output_bound_to_environment": True, "required_reports": ["MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json", "SOURCE_TRANSFER_REPORT.json"], "current_material_gate": True, "nested_bootstrap_key": "selected.material_gate.nested_bootstrap", "standardized_nested_bootstrap": True, "controller_native_robustness_bootstrap": True, "model_native_transport_head_bootstrap": True, "exact_entire_v69_fallback": True, "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"]},
            "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "output_written": False,
        }), indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        if args.support_probe_kr2:
            result = support_probe(dev, champion)
            result["preparation_affinity"] = affinity
            result["threadpools"] = threadpool_info()
            result["gpu_model_calls"] = 0
            print(json.dumps(clean(result), indent=2))
            return
        diagnostic, evidence, audits = run_nested(dev, champion, args.smoke_test)
        evaluation = evaluate(champion, diagnostic, evidence, audits, SMOKE_BOOTSTRAP_DRAWS if args.smoke_test else BOOTSTRAP_DRAWS, args.smoke_test)
    if args.smoke_test:
        print(json.dumps(clean({"status": "SMOKE_OK", "hypothesis": HYPOTHESIS, "preparation_affinity": affinity, "threadpools": threadpool_info(), "gpu_model_calls": 0, "folds": len(audits), "audits": audits, "outer_baseline": evaluation["outer_baseline"], "outer_candidate": evaluation["outer_candidate"], "outer_auc_delta": evaluation["outer_auc_delta"], "outer_ba_delta": evaluation["outer_ba_delta"], "outer_net_delta": evaluation["outer_net_delta"], "material_gate": current_material_gate(evaluation), "canonical_gate": evaluation["canonical_gate_audit"]["selected"], "fallback": evaluation["fallback"], "output_written": False}), indent=2))
        return
    result = write_outputs(args.output, authority, evidence, audits, evaluation)
    print(json.dumps(clean(result), indent=2))


if __name__ == "__main__":
    main()
