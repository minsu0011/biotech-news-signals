"""V155 fixed Random-Maclaurin degree-2/3 polynomial-kernel direction challenger.

Each embargoed train cut is transformed by the established train-only robust
causal numeric state (37 values plus 37 missingness indicators).  Equal event
groups are collapsed before a data-independent, seeded Rademacher Random
Maclaurin map: 256 degree-2 and 256 degree-3 projection products.  A single
fixed class-balanced ridge-logistic head is fitted on the train cut only.

This is not V84's sinusoidal shift-invariant RFF/Laplace model, V122's exact
eight-block Legendre chaos basis, or V154's data-dependent landmark RBF
Nyström/pairwise-AUC learner.  Inner-past labels choose exact V69 or one of
two predeclared blends.  Outer labels are evaluation-only.  Confidence and
high_conf remain exact V69; every failure restores the entire atomic V69
frame.  Audit/smoke/support are no-write, CPU30-31, two-thread, GPU-free.
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
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

import experiment_v149_multichannel_causal_increment_coskewness_tensor_direction as utility


base = utility.base
ROOT = utility.ROOT
V69_DIR = utility.V69_DIR
DEFAULT_OUT = ROOT / "research" / "staging" / "V155" / "LOCAL_FORBIDDEN"
VERSION = 155
HYPOTHESIS = "RANDOM_MACLAURIN_DEGREE23_POLYNOMIAL_KERNEL_DIRECTION_V1"
FEATURES = utility.FEATURES
EXPECTED_V69_EXPERIMENT_ID = utility.EXPECTED_V69_EXPERIMENT_ID
SCAFFOLD_PATH = ROOT / "experiment_v149_multichannel_causal_increment_coskewness_tensor_direction.py"
EXPECTED_SCAFFOLD_SHA256 = "4b9a364db95f598ebb75ca3406333412f1263ac1bb0722f68be81f7e17ab2fe0"

STATE_DIMENSION = 74
FEATURES_PER_DEGREE = 256
MACLAURIN_DIMENSION = 2 * FEATURES_PER_DEGREE
DEGREES = (2, 3)
PROJECTION_SCALE = 1.0 / np.sqrt(STATE_DIMENSION)
FEATURE_CLIP = 8.0
RIDGE_LOGISTIC_C = 0.50
SEED = 15501
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "RANDOM_MACLAURIN_DEGREE23_W0.25", "weight": 0.25},
    {"name": "RANDOM_MACLAURIN_DEGREE23_W0.50", "weight": 0.50},
)

require = utility.require
sha256 = utility.sha256
array_sha256 = utility.array_sha256
clean = utility.clean
bool_series = utility.bool_series
atomic_json = utility.atomic_json
atomic_csv = utility.atomic_csv
load_authorized = utility.load_authorized
metric = utility.metric
controller = utility.controller
v44 = utility.v44
numeric = utility.numeric
blend_probability = base.blend_probability

STATIC_SPEC = {
    "input": "74D train-cut robust causal numeric state (37 numeric plus 37 missing indicators)",
    "training_unit": "equal event_group centroid before nonlinear map",
    "degrees": list(DEGREES),
    "features_per_degree": FEATURES_PER_DEGREE,
    "feature_dimension": MACLAURIN_DIMENSION,
    "projection": "independent fixed seeded Rademacher matrices scaled by 1/sqrt(74)",
    "map": "degree-d product of d independent random linear projections, divided by sqrt(256)",
    "feature_scaling": "train-group-only coordinate median/IQR-or-std and fixed clip",
    "head": "one fixed equal-event-group/class-balanced C=0.5 ridge logistic",
    "data_dependent_landmark_frequency_or_basis_selection": False,
    "micro_tuning": False,
}
require(MACLAURIN_DIMENSION == 512 and DEGREES == (2, 3), "V155 fixed map changed")


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V149 utility scaffold changed")
    audit = utility.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable V36/V69 authority, chronology, robust numeric state, metrics, canonical gate, bootstrap and atomic IO only; no V149 output is read",
    }
    audit["v155_access_contract"] = {
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


class RandomMaclaurinDegree23:
    def __init__(self) -> None:
        generator = np.random.default_rng(SEED)
        self.projections: dict[int, tuple[np.ndarray, ...]] = {}
        for degree in DEGREES:
            self.projections[degree] = tuple(
                generator.choice((-1.0, 1.0), size=(STATE_DIMENSION, FEATURES_PER_DEGREE)).astype(float)
                * PROJECTION_SCALE
                for _ in range(degree)
            )
        flattened = np.concatenate([
            matrix.ravel() for degree in DEGREES for matrix in self.projections[degree]
        ])
        self.projection_inventory_sha256 = array_sha256(flattened)

    def transform(self, state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        require(state.ndim == 2 and state.shape[1] == STATE_DIMENSION, "V155 state dimension changed")
        branches = []
        branch_hashes: dict[str, str] = {}
        for degree in DEGREES:
            branch = np.ones((len(state), FEATURES_PER_DEGREE), dtype=float)
            for projection in self.projections[degree]:
                branch *= state @ projection
            branch /= np.sqrt(FEATURES_PER_DEGREE)
            require(np.isfinite(branch).all(), f"V155 degree-{degree} branch invalid")
            branches.append(branch)
            branch_hashes[str(degree)] = array_sha256(branch)
        feature = np.concatenate(branches, axis=1)
        require(feature.shape == (len(state), MACLAURIN_DIMENSION), "V155 map dimension changed")
        return feature, {
            "rows": len(feature),
            "feature_dimension": feature.shape[1],
            "degrees": list(DEGREES),
            "features_per_degree": FEATURES_PER_DEGREE,
            "seed": SEED,
            "projection_inventory_sha256": self.projection_inventory_sha256,
            "branch_sha256": branch_hashes,
            "feature_sha256": array_sha256(feature),
            "label_independent_fixed_projection": True,
            "random_maclaurin_projection_product_verified": True,
            "degree2_and_degree3_branches_verified": True,
            "data_dependent_landmark_frequency_or_basis_selection": False,
            "sinusoidal_random_fourier_feature": False,
            "semantic_block_legendre_chaos": False,
            "nystrom_rbf_or_pairwise_auc": False,
        }


class RobustFeatureScale:
    def __init__(self) -> None:
        self.median = np.empty(0, dtype=float)
        self.scale = np.empty(0, dtype=float)

    def fit(self, matrix: np.ndarray) -> "RobustFeatureScale":
        require(matrix.ndim == 2 and matrix.shape[1] == MACLAURIN_DIMENSION, "V155 scale shape changed")
        self.median = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(matrix, axis=0)
        self.scale = np.where(robust > 1e-8, robust, np.where(standard > 1e-8, standard, 1.0))
        require(np.isfinite(self.median).all() and np.isfinite(self.scale).all(), "V155 feature scale invalid")
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        result = np.clip((matrix - self.median) / self.scale, -FEATURE_CLIP, FEATURE_CLIP)
        require(np.isfinite(result).all(), "V155 scaled feature invalid")
        return result


def equal_event_group_state(
    ordered: pd.DataFrame,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    work = pd.DataFrame(state, columns=[f"state_{index}" for index in range(state.shape[1])])
    work["event_group_id"] = ordered.event_group_id.astype(str).to_numpy()
    work["y"] = ordered.y.to_numpy(int)
    label_n = work.groupby("event_group_id", sort=True).y.nunique()
    require(bool((label_n == 1).all()), "V155 event_group labels inconsistent")
    grouped = work.groupby("event_group_id", sort=True, as_index=False).agg(
        {**{f"state_{index}": "mean" for index in range(state.shape[1])}, "y": "first"}
    )
    group_state = grouped[[f"state_{index}" for index in range(state.shape[1])]].to_numpy(float)
    group_target = grouped.y.to_numpy(int)
    require(np.unique(group_target).size == 2, "V155 grouped train cut missing a class")
    return group_state, group_target, {
        "raw_rows": len(ordered),
        "equal_event_groups": len(grouped),
        "duplicate_rows_removed_before_nonlinear_map": len(ordered) - len(grouped),
        "class_event_group_n": np.bincount(group_target, minlength=2).tolist(),
        "equal_event_group_weighting": True,
    }


def random_maclaurin_direction(
    train: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(ordered.market.nunique() == 1 and target_frame.market.nunique() == 1, "V155 expects one market per fit")
    processor = numeric.RobustNumericState().fit(ordered)
    train_state, train_transform = processor.transform(ordered)
    target_state, target_transform = processor.transform(target_frame)
    require(train_state.shape[1] == target_state.shape[1] == STATE_DIMENSION, "V155 robust state must be 74D")
    group_state, group_target, grouping_audit = equal_event_group_state(ordered, train_state)
    mapper = RandomMaclaurinDegree23()
    group_feature, train_map_audit = mapper.transform(group_state)
    target_feature, target_map_audit = mapper.transform(target_state)
    feature_scale = RobustFeatureScale().fit(group_feature)
    train_design = feature_scale.transform(group_feature)
    target_design = feature_scale.transform(target_feature)
    class_n = np.bincount(group_target, minlength=2).astype(float)
    sample_weight = np.where(group_target == 1, 0.5 / class_n[1], 0.5 / class_n[0])
    head = LogisticRegression(
        C=RIDGE_LOGISTIC_C, penalty="l2", solver="liblinear",
        max_iter=1000, random_state=SEED,
    )
    head.fit(train_design, group_target, sample_weight=sample_weight)
    probability = np.clip(head.predict_proba(target_design)[:, 1], 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V155 probability invalid")
    return probability, {
        "train_n": len(ordered), "target_n": len(target_frame),
        "causal_numeric_only": True,
        "categorical_text_source_ticker_or_issuer_feature_n": 0,
        "train_transform": train_transform, "target_transform": target_transform,
        "grouping": grouping_audit,
        "static_spec": STATIC_SPEC,
        "train_kernel_map": train_map_audit,
        "target_kernel_map": target_map_audit,
        "feature_scaling": {
            "train_only": True, "clip": FEATURE_CLIP,
            "median_sha256": array_sha256(feature_scale.median),
            "scale_sha256": array_sha256(feature_scale.scale),
        },
        "head": {
            "type": "fixed equal-event-group/class-balanced pointwise ridge logistic",
            "objective": "pointwise balanced Bernoulli log loss plus fixed L2",
            "C": RIDGE_LOGISTIC_C, "solver": "liblinear",
            "iterations": int(head.n_iter_[0]),
            "class_event_group_n": class_n.astype(int).tolist(),
            "coefficient_l2": float(np.linalg.norm(head.coef_)),
            "coefficient_sha256": array_sha256(head.coef_),
            "intercept": head.intercept_.tolist(),
        },
        "target_rows_used_for_robust_scale_map_scale_or_head": False,
        "target_labels_used": False,
        "random_fourier_or_laplace_bayes": False,
        "semantic_block_legendre_polynomial_chaos": False,
        "nystrom_rbf_landmarks_or_pairwise_auc": False,
        "prediction": {
            "mean": float(probability.mean()), "std": float(probability.std()),
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
    challenger, model_audit = random_maclaurin_direction(inner_train, inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "architecture": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0, "all_trade_net_delta": 0.0,
        "worst_supported_source_ba_delta": 0.0, "supported_source_ba_delta": {},
        "eligible": True,
        "score": 2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"],
    }]
    model_eligible = bool(
        model_audit["causal_numeric_only"]
        and model_audit["grouping"]["equal_event_group_weighting"]
        and model_audit["train_kernel_map"]["random_maclaurin_projection_product_verified"]
        and model_audit["train_kernel_map"]["degree2_and_degree3_branches_verified"]
        and model_audit["target_kernel_map"]["random_maclaurin_projection_product_verified"]
        and model_audit["feature_scaling"]["train_only"]
        and not model_audit["target_rows_used_for_robust_scale_map_scale_or_head"]
        and not model_audit["target_labels_used"]
        and not model_audit["random_fourier_or_laplace_bayes"]
        and not model_audit["semantic_block_legendre_polynomial_chaos"]
        and not model_audit["nystrom_rbf_landmarks_or_pairwise_auc"]
    )
    for architecture in ARCHITECTURES:
        probability = blend_probability(baseline, challenger, architecture["weight"])
        current = metric(inner_valid, probability, confidence, high)
        auc_delta = current["auc"] - baseline_metric["auc"]
        ba_delta = current["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = current["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        source_floor, source_deltas = supported_source_ba_delta(inner_valid, baseline, probability)
        trials.append({
            "name": architecture["name"], "architecture": architecture, "metrics": current,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "worst_supported_source_ba_delta": source_floor,
            "supported_source_ba_delta": source_deltas,
            "model_eligible": model_eligible,
            "eligible": bool(model_eligible and ba_delta >= -0.005 and net_delta >= -0.0005 and source_floor >= -0.010),
            "score": 2.0 * current["auc"] + current["balanced_accuracy"],
        })
    selected = max(
        [trial for trial in trials if trial["eligible"]],
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 Random-Maclaurin degree-2/3 blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
        "projection_feature_count_seed_penalty_blend_or_source_floor_micro_tuning": False,
    }, model_audit


_BASE_BOOTSTRAP = base.paired_nested_bootstrap


def configure_base() -> None:
    base.VERSION = VERSION
    base.HYPOTHESIS = HYPOTHESIS
    base.STATIC_SPEC = STATIC_SPEC
    base.FRENET_FEATURE_DIMENSION = MACLAURIN_DIMENSION
    base.ARCHITECTURES = ARCHITECTURES
    base.SEED = SEED
    base.frenet_direction = random_maclaurin_direction


def run_nested(
    dev: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
    smoke_market: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    configure_base()
    base.choose_inner = choose_inner
    with redirect_stdout(StringIO()):
        diagnostic, evidence, audits = base.run_nested(dev, champion, smoke, smoke_market)
    diagnostic = diagnostic.rename(columns={"v146_model": "v155_model"})
    evidence = evidence.rename(columns={"frenet_prob": "random_maclaurin_prob", "v146_model": "v155_model"})
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V155 RANDOM MACLAURIN] market={audit['market']} fold={audit['fold']} "
            f"selected={selected['name']} inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audit['outer_auc_delta']:+.6f}",
            flush=True,
        )
    return diagnostic, evidence, audits


def paired_nested_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    configure_base()
    result = _BASE_BOOTSTRAP(evidence, draws)
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V155 Random-Maclaurin degree-2/3 policy"
    result["seed"] = SEED
    result["standardized"] = True
    return result


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
    require(candidate_summary["research_gate"] == canonical_candidate, "V155 canonical mismatch")
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
    maclaurin_contract = all(
        audit[side]["causal_numeric_only"]
        and audit[side]["grouping"]["equal_event_group_weighting"]
        and audit[side]["static_spec"]["feature_dimension"] == MACLAURIN_DIMENSION
        and audit[side]["train_kernel_map"]["feature_dimension"] == MACLAURIN_DIMENSION
        and audit[side]["train_kernel_map"]["random_maclaurin_projection_product_verified"]
        and audit[side]["train_kernel_map"]["degree2_and_degree3_branches_verified"]
        and audit[side]["target_kernel_map"]["random_maclaurin_projection_product_verified"]
        and audit[side]["feature_scaling"]["train_only"]
        and not audit[side]["target_rows_used_for_robust_scale_map_scale_or_head"]
        and not audit[side]["target_labels_used"]
        and not audit[side]["random_fourier_or_laplace_bayes"]
        and not audit[side]["semantic_block_legendre_polynomial_chaos"]
        and not audit[side]["nystrom_rbf_landmarks_or_pairwise_auc"]
        for audit in audits for side in ("inner_model_audit", "outer_model_audit")
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
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"] for audit in audits
        ),
        "random_maclaurin_degree23_polynomial_kernel_contract_verified": maclaurin_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected canonical mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V155 fallback not exact entire V69")
    return {
        "status": "SMOKE_DIAGNOSTIC_ONLY" if smoke else ("MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK"),
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": baseline, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - baseline["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - baseline["all_trade_mean_signed_net"],
        "nested_bootstrap": nested,
        "candidate_native_robustness_bootstrap": native,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline, "candidate": canonical_candidate,
            "selected": canonical_selected, "reported_equals_controller_recomputed": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact entire V69 DataFrame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or selected_frame.equals(champion)),
        },
        "diagnostic_frame": diagnostic_original, "selected_frame": selected_frame,
    }


def material_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    checks = evaluation["material_checks"]
    gate = {
        "contract": HYPOTHESIS, "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks), "material_pass": bool(evaluation["material_pass"]),
        "nested_bootstrap": evaluation["nested_bootstrap"],
        "native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        "current_hypothesis_not_inherited_from_v69": True,
        "fallback_policy": "candidate" if evaluation["material_pass"] else "exact entire V69 DataFrame",
    }
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V155 nested key mismatch")
    return gate


def enforce_output_conflict_fail_closed(out: Path) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V155 output requires controller authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V155 output target is not controller-bound")
    require(out.resolve().parent == (ROOT / "research" / "staging" / "V155").resolve(), "V155 output outside controller research/staging/V155")
    require(out.exists() and out.is_dir(), "V155 controller staging directory must already exist")
    required = {"VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log"}
    allowed = required | {"DATA_EPOCH_BINDING.json"}
    existing = {path.name for path in out.iterdir()}
    require(required.issubset(existing), f"V155 controller staging prefiles missing: {sorted(required - existing)}")
    require(not (existing - allowed), f"V155 unexpected pre-existing output conflict: {sorted(existing - allowed)}")
    return {
        "controller_bound": True, "target_preexisting": True,
        "required_controller_prefiles": sorted(required),
        "observed_controller_prefiles": sorted(existing), "unexpected_prefiles": [],
        "conflict_policy": "allow exact controller prefiles only; fail closed before any runner write",
    }


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT) and out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V155 output not controller-bound")
    conflict_audit = enforce_output_conflict_fail_closed(out)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require("bootstrap" in selected["robustness"], "V155 selected native bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V155", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V155_DIAGNOSTIC_RANDOM_MACLAURIN_DEGREE23": evaluation["candidate_summary"],
                "V155_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V155", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V155", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"], "seal_authorized": False,
        },
        "V155_RANDOM_MACLAURIN_DEGREE23_REPORT.json": {
            "version": "V155", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "STANDARDIZED_NESTED_BOOTSTRAP_REPORT.json": {
            "version": "V155", "hypothesis": HYPOTHESIS,
            "contract_key": "selected.material_gate.nested_bootstrap", "bootstrap": evaluation["nested_bootstrap"],
        },
        "NATIVE_ROBUSTNESS_BOOTSTRAP_REPORT.json": {
            "version": "V155", "hypothesis": HYPOTHESIS,
            "controller_native_robustness_bootstrap": evaluation["candidate_native_robustness_bootstrap"],
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V155", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": controller.canonical_research_gate(selected), "material_gate": gate,
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "seal_state": "UNOPENED", "seal_authorized": False,
            "output_conflict_audit": conflict_audit,
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V155_RANDOM_MACLAURIN_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V155_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
    modes.add_argument("--support-probe", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--smoke-market", choices=("US", "KR"), default="US")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        require(
            not args.audit_only and not args.smoke_test and not args.support_probe
            and not args.full_run and args.output is None,
            "explicit mode/output conflicts with controller-bound no-argument V155 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V155 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V155 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
    runtime = numeric.configure_bounded_runtime() if bounded else {
        "mode": "controller_bound_full", "thread_policy": "authorized parent inherited", "gpu_model_calls": 0,
    }
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        mapper = RandomMaclaurinDegree23()
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "runtime": runtime,
            "dev_rows": len(dev), "v69_rows": len(champion),
            "features": FEATURES, "static_spec": STATIC_SPEC,
            "projection_inventory_sha256": mapper.projection_inventory_sha256,
            "architecture": "fixed seeded 256 degree-2 plus 256 degree-3 Rademacher Random-Maclaurin products and one equal-group balanced ridge logistic",
            "causal_numeric_only": True, "target_row_labels_used": False,
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
                "native_robustness_bootstrap_preserved": True,
                "source_transfer_current_gate": True,
                "exact_entire_v69_fallback": True,
                "authorized_prefiles": ["VERSION_PLAN.json", "BOTTLENECK_ANALYSIS.json", "MATERIAL_CHANGE.json", "RUN.log", "DATA_EPOCH_BINDING.json"],
            },
            "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    with threadpool_limits(limits=SMOKE_THREADS if bounded else None):
        diagnostic, evidence, audits = run_nested(dev, champion, smoke=bounded, smoke_market=args.smoke_market)
        if args.support_probe:
            require(args.smoke_market == "KR", "V155 support probe is reserved for KR:2")
            audit = audits[0]
            non_noop = [trial for trial in audit["policy"]["trials"] if trial["architecture"] is not None]
            best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
            print(json.dumps(clean({
                "status": "SUPPORT_PROBE_OK", "hypothesis": HYPOTHESIS,
                "runtime": runtime, "market": "KR", "fold": 2,
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
        "hypothesis": HYPOTHESIS, "runtime": runtime,
        "folds_executed": len(audits), "smoke_market": args.smoke_market if args.smoke_test else None,
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
