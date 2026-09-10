"""V123 US_SEC diagonal-GMM Fisher-vector direction challenger.

For every embargoed US past cut, only historical US_SEC rows fit a deterministic
five-component duplicate-weighted diagonal Gaussian mixture on the immutable 37
robust causal numeric fields.  The label-free mixture turns each row into the
classical Fisher score with respect to mixture weights, means, and log variances;
signed-square-root and L2 normalization yield a fixed 375-dimensional vector.
A fixed duplicate/class-balanced ridge-logistic head maps the vector to direction.

Only US_SEC rows may change.  US_NEWS and every KR row remain exact V69.  Inner
US_SEC labels select exact V69 or fixed .25/.50 blends; outer labels remain
evaluation-only with a strict 35-minute embargo.  V69 confidence/high-confidence
are frozen.  Material failure restores the exact entire V69 DataFrame.

This is not class-conditional QDA/density fitting, RFF/kernel learning, ICA/NMF/
PLS, an archetype/prototype, quantum density, RQA, HDC, polynomial chaos, tree,
or neural representation.  Audit and US:2 smoke are CPU30-31/two-thread/no-write/
no-GPU.  Full execution is accepted only from no-argument controller binding via
MARKET_BIO_VERSION_OUTPUT and writes the complete current-gate report contract.
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
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

import experiment_v119_per_row_ridge_koopman_dmd_spectral as scaffold


ROOT = scaffold.ROOT
V69_DIR = scaffold.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V123_US_SEC_DIAGONAL_GMM_FISHER_VECTOR_V1"
VERSION = 123
HYPOTHESIS = "US_SEC_DIAGONAL_GMM_FISHER_VECTOR_DIRECTION_V1"
FEATURES = scaffold.FEATURES
EXPECTED_MARKET_FOLD_ROWS = scaffold.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v119_per_row_ridge_koopman_dmd_spectral.py"
EXPECTED_SCAFFOLD_SHA256 = "c8d23ea87997d1f669efd77316ceb0b5b99b9a45fae945cf21ff9588f11b2cc6"

SOURCE_SCOPE = "US_SEC"
GMM_COMPONENTS = 5
GMM_ITERATIONS = 48
VARIANCE_FLOOR = 0.05
MIXTURE_FLOOR = 1e-4
FISHER_FEATURE_DIMENSION = GMM_COMPONENTS * (1 + 2 * len(FEATURES))
RIDGE_LOGISTIC_C = 0.50
SEED = 12301
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "US_SEC_GMM_FISHER_W0.25", "weight": 0.25},
    {"name": "US_SEC_GMM_FISHER_W0.50", "weight": 0.50},
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


def verify_static_spec() -> dict[str, Any]:
    require(GMM_COMPONENTS == 5 and GMM_ITERATIONS == 48, "V123 fixed GMM specification changed")
    require(FISHER_FEATURE_DIMENSION == 375, "V123 Fisher-vector dimension changed")
    return {
        "source_scope": SOURCE_SCOPE,
        "numeric_feature_n": len(FEATURES),
        "gmm_components": GMM_COMPONENTS,
        "gmm_covariance": "diagonal",
        "gmm_iterations": GMM_ITERATIONS,
        "variance_floor": VARIANCE_FLOOR,
        "mixture_floor": MIXTURE_FLOOR,
        "fisher_terms": ["mixture_weight", "mean", "log_variance"],
        "fisher_feature_dimension": FISHER_FEATURE_DIMENSION,
        "fisher_normalization": "signed square root then row L2",
        "non_scope_policy": "exact V69",
    }


STATIC_SPEC = verify_static_spec()


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V119 utility scaffold changed")
    audit = scaffold.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, robust state, metrics, canonical gate, blending, and atomic IO only; no V119 output is read",
    }
    audit["v123_access_contract"] = {
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


def duplicate_unit_weights(frame: pd.DataFrame) -> np.ndarray:
    group_size = frame.groupby("event_group_id", sort=False).event_id.transform("size").to_numpy(float)
    weight = 1.0 / np.maximum(group_size, 1.0)
    require(np.isfinite(weight).all() and np.all(weight > 0.0), "V123 duplicate weight invalid")
    return weight


def logsumexp(matrix: np.ndarray) -> np.ndarray:
    maximum = np.max(matrix, axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.sum(np.exp(matrix - maximum), axis=1))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    require(float(cumulative[-1]) > 0.0, "V123 weighted quantile zero mass")
    position = int(np.searchsorted(cumulative, quantile * float(cumulative[-1]), side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


class DiagonalGmm:
    def __init__(self) -> None:
        self.mixture = np.empty(0, dtype=float)
        self.mean = np.empty((0, 0), dtype=float)
        self.variance = np.empty((0, 0), dtype=float)
        self.audit: dict[str, Any] = {}

    def component_log_probability(self, matrix: np.ndarray) -> np.ndarray:
        difference = matrix[:, None, :] - self.mean[None, :, :]
        log_density = -0.5 * np.sum(
            np.log(2.0 * np.pi * self.variance[None, :, :])
            + difference * difference / self.variance[None, :, :],
            axis=2,
        )
        return log_density + np.log(self.mixture[None, :])

    def responsibilities(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_joint = self.component_log_probability(matrix)
        normalizer = logsumexp(log_joint)
        responsibility = np.exp(log_joint - normalizer[:, None])
        require(np.isfinite(responsibility).all(), "V123 GMM responsibility invalid")
        require(float(np.max(np.abs(responsibility.sum(axis=1) - 1.0))) <= 1e-10, "V123 responsibility normalization failed")
        return responsibility, normalizer

    def fit(self, matrix: np.ndarray, sample_weight: np.ndarray) -> "DiagonalGmm":
        require(matrix.ndim == 2 and matrix.shape[1] == len(FEATURES), "V123 GMM matrix shape changed")
        require(len(matrix) >= 10 * GMM_COMPONENTS, "V123 GMM support too small")
        weight = sample_weight / float(sample_weight.sum())
        projection = np.mean(matrix, axis=1)
        anchors = [weighted_quantile(projection, weight, (component + 0.5) / GMM_COMPONENTS) for component in range(GMM_COMPONENTS)]
        chosen: list[int] = []
        for anchor in anchors:
            ordering = np.argsort(np.abs(projection - anchor), kind="stable")
            chosen.append(next(int(index) for index in ordering if int(index) not in chosen))
        self.mean = matrix[np.asarray(chosen, dtype=int)].copy()
        global_mean = np.sum(weight[:, None] * matrix, axis=0)
        global_variance = np.sum(weight[:, None] * (matrix - global_mean) ** 2, axis=0)
        global_variance = np.maximum(global_variance, VARIANCE_FLOOR)
        self.variance = np.broadcast_to(global_variance[None, :], (GMM_COMPONENTS, matrix.shape[1])).copy()
        self.mixture = np.full(GMM_COMPONENTS, 1.0 / GMM_COMPONENTS, dtype=float)
        trace: list[float] = []
        for _ in range(GMM_ITERATIONS):
            responsibility, log_likelihood = self.responsibilities(matrix)
            trace.append(float(np.sum(weight * log_likelihood)))
            weighted_responsibility = weight[:, None] * responsibility
            mass = weighted_responsibility.sum(axis=0)
            require(np.all(mass > 1e-10), "V123 GMM component collapsed")
            self.mixture = np.maximum(mass, MIXTURE_FLOOR)
            self.mixture /= float(self.mixture.sum())
            self.mean = (weighted_responsibility.T @ matrix) / mass[:, None]
            difference = matrix[:, None, :] - self.mean[None, :, :]
            self.variance = np.sum(weighted_responsibility[:, :, None] * difference * difference, axis=0) / mass[:, None]
            self.variance = np.maximum(self.variance, VARIANCE_FLOOR)
        responsibility, log_likelihood = self.responsibilities(matrix)
        final_value = float(np.sum(weight * log_likelihood))
        trace.append(final_value)
        self.audit = {
            "fit_rows": len(matrix),
            "components": GMM_COMPONENTS,
            "iterations": GMM_ITERATIONS,
            "initial_weighted_mean_log_likelihood": trace[0],
            "final_weighted_mean_log_likelihood": final_value,
            "minimum_mixture_weight": float(self.mixture.min()),
            "maximum_mixture_weight": float(self.mixture.max()),
            "minimum_variance": float(self.variance.min()),
            "maximum_variance": float(self.variance.max()),
            "mixture_sha256": array_sha256(self.mixture),
            "mean_sha256": array_sha256(self.mean),
            "variance_sha256": array_sha256(self.variance),
            "responsibility_entropy_mean": float(np.mean(-np.sum(responsibility * np.log(np.maximum(responsibility, 1e-15)), axis=1))),
            "label_free_fit": True,
            "duplicate_weighted": True,
        }
        return self


def fisher_vector(model: DiagonalGmm, matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    responsibility, _ = model.responsibilities(matrix)
    mixture = np.maximum(model.mixture, MIXTURE_FLOOR)
    difference = matrix[:, None, :] - model.mean[None, :, :]
    standard = np.sqrt(model.variance)[None, :, :]
    weight_score = (responsibility - mixture[None, :]) / np.sqrt(mixture[None, :])
    mean_score = responsibility[:, :, None] * (difference / standard) / np.sqrt(mixture[None, :, None])
    variance_score = responsibility[:, :, None] * (
        difference * difference / model.variance[None, :, :] - 1.0
    ) / np.sqrt(2.0 * mixture[None, :, None])
    raw = np.concatenate([
        weight_score,
        mean_score.reshape(len(matrix), -1),
        variance_score.reshape(len(matrix), -1),
    ], axis=1)
    require(raw.shape[1] == FISHER_FEATURE_DIMENSION, "V123 Fisher dimension mismatch")
    powered = np.sign(raw) * np.sqrt(np.abs(raw))
    norm = np.linalg.norm(powered, axis=1, keepdims=True)
    feature = powered / np.maximum(norm, 1e-12)
    require(np.isfinite(feature).all(), "V123 Fisher vector invalid")
    return feature, {
        "rows": len(feature),
        "feature_dimension": feature.shape[1],
        "raw_l2_mean": float(np.mean(np.linalg.norm(raw, axis=1))),
        "normalized_l2_min": float(np.min(np.linalg.norm(feature, axis=1))),
        "normalized_l2_max": float(np.max(np.linalg.norm(feature, axis=1))),
        "responsibility_max_mean": float(np.mean(np.max(responsibility, axis=1))),
        "feature_sha256": array_sha256(feature),
        "signed_square_root": True,
        "row_l2_normalized": True,
    }


def fisher_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.eq("US").all() and target_frame.market.eq("US").all(), "V123 expects US only")
    require(ordered.source_family.eq(SOURCE_SCOPE).all(), "V123 train source escaped US_SEC")
    require(target_frame.source_family.eq(SOURCE_SCOPE).all(), "V123 target source escaped US_SEC")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    train_numeric = train_state[:, :len(FEATURES)]
    target_numeric = target_state[:, :len(FEATURES)]
    duplicate_weight = duplicate_unit_weights(ordered)
    mixture = DiagonalGmm().fit(train_numeric, duplicate_weight)
    train_feature, train_fisher_audit = fisher_vector(mixture, train_numeric)
    target_feature, target_fisher_audit = fisher_vector(mixture, target_numeric)
    target = ordered.y.to_numpy(int)
    head_weight = numeric.duplicate_class_weights(ordered)
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        random_state=SEED,
        n_jobs=1,
    )
    head.fit(train_feature, target, sample_weight=head_weight)
    probability = np.clip(head.predict_proba(target_feature)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V123 probability invalid")
    return probability, {
        "scope_applicable": True,
        "train_n": len(ordered),
        "target_n": len(target_frame),
        "source_scope": SOURCE_SCOPE,
        "causal_numeric_only": True,
        "categorical_text_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform,
        "target_transform": target_transform,
        "static_spec": STATIC_SPEC,
        "gmm": mixture.audit,
        "train_fisher": train_fisher_audit,
        "target_fisher": target_fisher_audit,
        "head": {
            "type": "fixed duplicate/class-balanced ridge logistic",
            "C": RIDGE_LOGISTIC_C,
            "class_n": np.bincount(target, minlength=2).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_robust_scale_gmm_or_head": False,
        "target_labels_used": False,
        "gmm_labels_used": False,
        "class_conditional_gmm_or_qda": False,
        "kernel_rff_tree_or_neural": False,
        "ica_nmf_pls_or_archetype": False,
        "quantum_density_or_born_fidelity": False,
        "rqa_hdc_or_polynomial_chaos": False,
        "prediction": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "probability_sha256": array_sha256(probability),
        },
    }


def subset_source(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.market.eq("US") & frame.source_family.eq(SOURCE_SCOPE)].copy()


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    inner_champion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_scope = subset_source(inner_train)
    valid_mask = inner_valid.market.eq("US") & inner_valid.source_family.eq(SOURCE_SCOPE)
    valid_scope = inner_valid.loc[valid_mask].copy()
    champion_scope = inner_champion.loc[valid_mask.to_numpy()].copy()
    require(len(valid_scope) == len(champion_scope) >= 100, "V123 inner US_SEC support changed")
    challenger, model_audit = fisher_direction(train_scope, valid_scope)
    baseline = champion_scope.prob.to_numpy(float)
    confidence = champion_scope.confidence_signal.to_numpy(float)
    high = bool_series(champion_scope.high_conf).to_numpy(bool)
    baseline_metric = metric(valid_scope, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP",
        "architecture": None,
        "metrics": baseline_metric,
        "auc_delta": 0.0,
        "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0,
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(valid_scope, probability, confidence, high)
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
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
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
        "selection_scope": SOURCE_SCOPE,
        "selection_rule": "inner-past US_SEC OOF only; exact V69 no-op or fixed .25/.50 GMM-Fisher blend; fixed 2*AUC+BA with BA/net eligibility",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "component_iteration_floor_head_or_blend_micro_tuning": False,
    }, model_audit


def noop_policy(inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> dict[str, Any]:
    baseline = metric(
        inner_valid,
        inner_champion.prob.to_numpy(float),
        inner_champion.confidence_signal.to_numpy(float),
        bool_series(inner_champion.high_conf).to_numpy(bool),
    )
    trial = {
        "name": "V69_NOOP",
        "architecture": None,
        "metrics": baseline,
        "auc_delta": 0.0,
        "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0,
        "eligible": True,
        "score": 2.0 * baseline["auc"] + baseline["balanced_accuracy"],
    }
    return {
        "selection_scope": "OUTSIDE_US_SEC_FORCED_EXACT_V69",
        "selection_rule": "non-US market is outside immutable V123 scope",
        "baseline": baseline,
        "selected": trial,
        "trials": [trial],
        "outer_labels_used_for_selection": False,
    }


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v123_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in (fold_specs[:1] if smoke else fold_specs):
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} size changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, market, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V123 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        challenger_full = baseline.copy()
        if market == "US":
            policy, inner_model_audit = choose_inner(inner_train, inner_valid, inner_champion)
            outer_scope_mask = outer_valid.source_family.eq(SOURCE_SCOPE).to_numpy(bool)
            outer_train_scope = subset_source(outer_train)
            outer_valid_scope = outer_valid.loc[outer_scope_mask].copy()
            require(len(outer_valid_scope) >= 100, "V123 outer US_SEC support changed")
            challenger_scope, outer_model_audit = fisher_direction(outer_train_scope, outer_valid_scope)
            challenger_full[outer_scope_mask] = challenger_scope
        else:
            policy = noop_policy(inner_valid, inner_champion)
            inner_model_audit = {"scope_applicable": False, "reason": "KR outside US_SEC scope"}
            outer_model_audit = {"scope_applicable": False, "reason": "KR outside US_SEC scope"}
            outer_scope_mask = np.zeros(len(outer_valid), dtype=bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["architecture"] is None else blend_probability(
            baseline, challenger_full, selected["architecture"]["weight"],
        )
        require(np.array_equal(candidate[~outer_scope_mask], baseline[~outer_scope_mask]), "V123 changed out-of-scope row")
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v123_model"] = np.where(outer_scope_mask, selected["name"], "V69_NOOP")
        outer_diagnostics: dict[str, Any] = {}
        if market == "US":
            scoped_frame = outer_valid.loc[outer_scope_mask]
            scoped_baseline = baseline[outer_scope_mask]
            scoped_challenger = challenger_full[outer_scope_mask]
            scoped_confidence = confidence[outer_scope_mask]
            scoped_high = high[outer_scope_mask]
            outer_diagnostics = {
                architecture["name"]: metric(
                    scoped_frame,
                    blend_probability(scoped_baseline, scoped_challenger, architecture["weight"]),
                    scoped_confidence,
                    scoped_high,
                ) for architecture in ARCHITECTURES
            }
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        scope_base = metric(
            outer_valid.loc[outer_scope_mask], baseline[outer_scope_mask],
            confidence[outer_scope_mask], high[outer_scope_mask],
        ) if outer_scope_mask.any() else None
        scope_candidate = metric(
            outer_valid.loc[outer_scope_mask], candidate[outer_scope_mask],
            confidence[outer_scope_mask], high[outer_scope_mask],
        ) if outer_scope_mask.any() else None
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["scope_applicable"] = outer_scope_mask
        evidence["baseline_prob"] = baseline
        evidence["fisher_prob"] = challenger_full
        evidence["candidate_prob"] = candidate
        evidence["v123_model"] = np.where(outer_scope_mask, selected["name"], "V69_NOOP")
        evidence_parts.append(evidence)
        audits.append({
            "market": market,
            "fold": fold,
            "scope_applicable": market == "US",
            "inner_chronology": inner_chronology,
            "outer_chronology": outer_chronology,
            "policy": policy,
            "inner_model_audit": inner_model_audit,
            "outer_model_audit": outer_model_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric,
            "outer_candidate": candidate_metric,
            "outer_scope_baseline": scope_base,
            "outer_scope_candidate": scope_candidate,
            "outer_scope_auc_delta": None if scope_base is None else scope_candidate["auc"] - scope_base["auc"],
            "outer_scope_ba_delta": None if scope_base is None else scope_candidate["balanced_accuracy"] - scope_base["balanced_accuracy"],
            "outer_scope_net_delta": None if scope_base is None else scope_candidate["all_trade_mean_signed_net"] - scope_base["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V123 US_SEC GMM-FISHER] market={market} fold={fold} selected={selected['name']} "
            f"scope_auc_delta={audits[-1]['outer_scope_auc_delta']}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V123 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V123 changed confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V123 changed high_conf")
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.loc[evidence.scope_applicable].copy()
    require(len(work) >= 100, "V123 scoped bootstrap support too small")
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = timestamp.dt.strftime("US|%Y-%m")
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V123 nested-bootstrap blocks")
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
    require(len(values["auc_delta"]) >= int(0.90 * draws), "V123 bootstrap lost too many draws")

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
        "scope": SOURCE_SCOPE,
        "method": "paired US_SEC fold-month block bootstrap over inner-locked V123 GMM-Fisher policy",
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V123 canonical mismatch")
    scoped = evidence.loc[evidence.scope_applicable].copy()
    baseline = metric(scoped, scoped.baseline_prob, scoped.confidence_signal, bool_series(scoped.high_conf))
    candidate = metric(scoped, scoped.candidate_prob, scoped.confidence_signal, bool_series(scoped.high_conf))
    nested_bootstrap = paired_nested_bootstrap(evidence, draws)
    applicable = [audit for audit in audits if audit["scope_applicable"]]
    fold_auc = np.asarray([audit["outer_scope_auc_delta"] for audit in applicable], float)
    fold_net = np.asarray([audit["outer_scope_net_delta"] for audit in applicable], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = (
        np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float))
        and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    )
    fisher_contract = bool(applicable) and all(
        audit[side]["scope_applicable"]
        and audit[side]["source_scope"] == SOURCE_SCOPE
        and audit[side]["causal_numeric_only"]
        and audit[side]["gmm"]["components"] == GMM_COMPONENTS
        and audit[side]["gmm"]["iterations"] == GMM_ITERATIONS
        and audit[side]["gmm"]["label_free_fit"]
        and audit[side]["gmm"]["duplicate_weighted"]
        and audit[side]["train_fisher"]["feature_dimension"] == FISHER_FEATURE_DIMENSION
        and audit[side]["target_fisher"]["signed_square_root"]
        and audit[side]["target_fisher"]["row_l2_normalized"]
        and not audit[side]["target_rows_used_for_robust_scale_gmm_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["gmm_labels_used"]
        and not audit[side]["class_conditional_gmm_or_qda"]
        and not audit[side]["kernel_rff_tree_or_neural"]
        and not audit[side]["ica_nmf_pls_or_archetype"]
        and not audit[side]["quantum_density_or_born_fidelity"]
        and not audit[side]["rqa_hdc_or_polynomial_chaos"]
        for audit in applicable for side in ("inner_model_audit", "outer_model_audit")
    )
    out_of_scope_exact = np.array_equal(
        evidence.loc[~evidence.scope_applicable, "candidate_prob"].to_numpy(float),
        evidence.loc[~evidence.scope_applicable, "baseline_prob"].to_numpy(float),
    )
    native_bootstrap = candidate_summary["robustness"]["bootstrap"]
    required_native = {
        "balanced_accuracy_lower95", "balanced_accuracy_upper95",
        "highconf_strategy_net_lower95", "highconf_strategy_net_upper95",
    }
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "us_sec_outer_auc_delta_gt_0_003": candidate["auc"] - baseline["auc"] > 0.003,
        "us_sec_outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - baseline["balanced_accuracy"] > 0.0,
        "us_sec_outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"] >= 0.0,
        "positive_us_sec_outer_fold_auc_fraction_ge_2_of_3": len(fold_auc) == 3 and float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_us_sec_outer_fold_net_fraction_ge_2_of_3": len(fold_net) == 3 and float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "nested_bootstrap_auc_probability_gt_zero_ge_0_75": nested_bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "nested_bootstrap_net_probability_gt_zero_ge_0_65": nested_bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "candidate_native_robustness_bootstrap_keys": required_native.issubset(native_bootstrap),
        "v69_confidence_highconf_and_out_of_scope_exact": confidence_exact and out_of_scope_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "us_sec_diagonal_gmm_fisher_vector_contract_verified": fisher_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V123 fallback is not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else (
            "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"
        ),
        "material_pass": material_pass,
        "material_checks": checks,
        "scope": SOURCE_SCOPE,
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
        "scope": SOURCE_SCOPE,
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V123 nested key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V123 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V123 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V123",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V123_DIAGNOSTIC_US_SEC_GMM_FISHER": evaluation["candidate_summary"],
                "V123_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact,
            "authority_audit": authority,
            "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V123",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "opened_dev_only": True,
            "selected": selected,
            "candidate": evaluation["candidate_summary"],
            "material_gate": gate,
            "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V123",
            "hypothesis": HYPOTHESIS,
            "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate,
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V123_US_SEC_DIAGONAL_GMM_FISHER_VECTOR_REPORT.json": {
            "version": "V123",
            "hypothesis": HYPOTHESIS,
            "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES,
            "nested_fold_audits": audits,
            "evaluation": compact,
            "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V123",
            "status": "MATCH",
            "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V123_US_SEC_GMM_FISHER_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V123_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V123 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V123 direct full run forbidden; MARKET_BIO_VERSION_OUTPUT required")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "controller full run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    runtime = numeric.configure_bounded_runtime() if (args.audit_only or args.smoke_test) else {
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
            "architecture": "US_SEC-only duplicate-weighted label-free diagonal-GMM Fisher vector plus fixed balanced ridge-logistic head",
            "causal_numeric_only": True,
            "target_row_labels_used": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "research_staging_v123_compatible": True,
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
    with threadpool_limits(limits=SMOKE_THREADS if args.smoke_test else None):
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
        "outer_scope": SOURCE_SCOPE,
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
        summary["raw_outer_scope_baseline"] = audits[0]["outer_scope_baseline"]
        summary["raw_outer_scope_selected"] = audits[0]["outer_scope_candidate"]
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
