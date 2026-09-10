"""V69 fail-closed US_SEC structured-agreement confidence selector.

V58 direction probabilities and predictions are immutable.  V59's structured
probability is reduced to a direction-agreement bit and may affect only the
confidence/opportunity ordering of US_SEC rows.  A small predeclared bonus grid
is selected on embargoed inner-past rows, then evaluated on untouched outer
chronological rows while preserving the V58 US_SEC high-confidence count in
each outer block.

All material gates are predeclared below.  Failure returns the exact V58
probability, confidence_signal, and high_conf arrays.  This runner reads only
the two pinned DEV OOF files and writes only its staging directory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
V58_PATH = CACHE / "v58_locked_source_routing_oof.csv.gz"
V59_PATH = CACHE / "v59_sec_structured_oof.csv.gz"
OUT = Path(
    os.environ.get(
        "MARKET_BIO_VERSION_OUTPUT",
        str(ROOT / "staging" / "V69_US_SEC_STRUCTURED_AGREEMENT_SELECTOR_V1"),
    )
)
HYPOTHESIS = "US_SEC_STRUCTURED_AGREEMENT_SELECTOR_V1"
VERSION = 69
EXPECTED_SHA256 = {
    V58_PATH: "8509a953f499ed4b9d9eebf33ff9ecb8909edb6bd41a8eb974f6d3963d4c665c",
    V59_PATH: "13260dbe381943dd1f2d4e7d30d6e1301c7fea159b1748ab1d40836f6e7e7dd4",
}
EXPECTED_ROWS = 7568
EXPECTED_US_SEC_ROWS = 5426
BONUS_GRID = (0.00, 0.05, 0.10, 0.15, 0.20)
EMBARGO = pd.Timedelta(minutes=35)
FIRST_OUTER_BURN_IN_FRACTION = 0.70
INNER_VALID_FRACTION = 0.30
COST = 0.002
SEED = 6901
BOOTSTRAP_DRAWS = 5000
HC_ACCURACY_REGRESSION_FLOOR = -0.005
CONFIDENCE_AUC_REGRESSION_FLOOR = -0.005
BOOTSTRAP_POSITIVE_PROBABILITY_FLOOR = 0.85
POSITIVE_FOLD_FRACTION_FLOOR = 0.75
OVERALL_COVERAGE_FLOOR = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(target, score)) if np.unique(target).size == 2 else None


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    raw = (json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(raw, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.name.endswith(".gz"):
        with temporary.open("wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
                compressed.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary.write_bytes(raw)
    os.replace(temporary, path)


def verify_and_load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"required V69 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V69 input changed: {path.name}")
        hashes[str(path.relative_to(ROOT))] = actual
    v58 = pd.read_csv(V58_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    v59 = pd.read_csv(V59_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    require(len(v58) == EXPECTED_ROWS and len(v59) == EXPECTED_ROWS, "OOF row-count contract changed")
    require(v58.event_id.is_unique and v59.event_id.is_unique, "OOF event_id is not unique")
    require(set(v58.event_id) == set(v59.event_id), "V58/V59 event sets differ")
    v59 = v59.set_index("event_id").loc[v58.event_id].reset_index()
    for column in ("event_group_id", "market", "ticker", "source_family", "form", "y", "fold"):
        require(v58[column].astype(str).equals(v59[column].astype(str)), f"V58/V59 {column} differs")
    require(
        np.allclose(v58.fwd_ret_30m.to_numpy(float), v59.fwd_ret_30m.to_numpy(float), rtol=0.0, atol=0.0),
        "V58/V59 return labels differ",
    )
    require(np.isfinite(v58.prob).all() and np.isfinite(v59.prob).all(), "non-finite direction probability")
    require(v58.prob.between(0.0, 1.0).all() and v59.prob.between(0.0, 1.0).all(), "probability outside [0,1]")
    non_sec = v58.source_family.ne("US_SEC")
    require(
        np.array_equal(v58.loc[non_sec, "prob"].to_numpy(float), v59.loc[non_sec, "prob"].to_numpy(float)),
        "V59 changed a non-US_SEC direction probability",
    )
    us_sec = v58.source_family.eq("US_SEC")
    require(int(us_sec.sum()) == EXPECTED_US_SEC_ROWS, "US_SEC OOF row-count contract changed")
    require(v58.loc[us_sec, "market"].eq("US").all(), "US_SEC contains a non-US row")
    v58["event_time_utc"] = pd.to_datetime(v58.event_time_utc, utc=True)
    v59["event_time_utc"] = pd.to_datetime(v59.event_time_utc, utc=True)
    audit = {
        "input_sha256": hashes,
        "authorized_inputs": [str(V58_PATH.relative_to(ROOT)), str(V59_PATH.relative_to(ROOT))],
        "rows": len(v58),
        "us_sec_rows": int(us_sec.sum()),
        "dev_oof_only": True,
        "dev_extension_loaded": False,
        "research_seal_pool_loaded": False,
        "final_meta_reserve_loaded": False,
        "v59_probability_authorized_use": "threshold at 0.5 to form agreement; confidence/opportunity ranking only",
        "v59_probability_used_for_direction_output": False,
        "raw_feature_sign_flip_allowed": False,
        "raw_feature_sign_flip_performed": False,
    }
    return v58, v59, audit


def add_agreement(v58: pd.DataFrame, v59: pd.DataFrame) -> pd.DataFrame:
    sec = v58.loc[v58.source_family.eq("US_SEC")].copy()
    structured_probability = v59.set_index("event_id").loc[sec.event_id, "prob"].to_numpy(float)
    sec["structured_prob"] = structured_probability
    sec["v58_direction"] = sec.prob.to_numpy(float) >= 0.5
    sec["v59_structured_direction"] = structured_probability >= 0.5
    sec["structured_agreement"] = sec.v58_direction.eq(sec.v59_structured_direction)
    sec["direction_correct"] = sec.v58_direction.eq(sec.y.astype(bool))
    sec["signed_gross"] = np.where(sec.v58_direction, 1.0, -1.0) * sec.fwd_ret_30m.to_numpy(float)
    sec["signed_net"] = sec.signed_gross - COST
    sec["_base_row"] = sec.index.to_numpy(int)
    sec = sec.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    sec["selector_confidence_candidate"] = sec.confidence_signal.to_numpy(float)
    sec["selector_high_candidate"] = sec.high_conf.to_numpy(bool)
    sec["selector_bonus"] = 0.0
    sec["selector_outer_fold"] = 0
    sec["selector_evidence_eligible"] = False
    return sec


def candidate_score(frame: pd.DataFrame, bonus: float) -> np.ndarray:
    """Bounded score; rank-equivalent to both bonus and penalty forms.

    (base + b*agreement)/(1+b) has the same within-block ordering as
    base + b*agreement and as base - b*disagreement.  The denominator keeps
    confidence in [0,1], and b=0 is an exact V58 no-op.
    """
    base = frame.confidence_signal.to_numpy(float)
    agreement = frame.structured_agreement.to_numpy(float)
    score = (base + float(bonus) * agreement) / (1.0 + float(bonus))
    require(np.isfinite(score).all() and np.logical_and(score >= 0.0, score <= 1.0).all(), "invalid candidate score")
    return score


def stable_top_k(frame: pd.DataFrame, score: np.ndarray, k: int) -> np.ndarray:
    require(0 <= k <= len(frame), "invalid high-confidence target count")
    selected = np.zeros(len(frame), dtype=bool)
    if k:
        order = np.lexsort((frame.event_id.astype(str).to_numpy(), -np.asarray(score, float)))
        selected[order[:k]] = True
    return selected


def apply_policy(frame: pd.DataFrame, bonus: float) -> tuple[np.ndarray, np.ndarray]:
    score = candidate_score(frame, bonus)
    baseline = frame.high_conf.to_numpy(bool)
    if float(bonus) == 0.0:
        selected = baseline.copy()
    else:
        selected = stable_top_k(frame, score, int(baseline.sum()))
    require(int(selected.sum()) == int(baseline.sum()), "US_SEC high-confidence count changed")
    return score, selected


def policy_metrics(frame: pd.DataFrame, score: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    high = np.asarray(high, bool)
    correct = frame.direction_correct.to_numpy(bool)
    net = frame.signed_net.to_numpy(float)
    gross = frame.signed_gross.to_numpy(float)
    weights = frame.cluster_weight.to_numpy(float)
    weighted_denominator = float(weights[high].sum()) if high.any() else 0.0
    return {
        "n": len(frame),
        "highconf_n": int(high.sum()),
        "highconf_coverage": float(high.mean()),
        "highconf_accuracy": float(correct[high].mean()) if high.any() else None,
        "highconf_mean_signed_gross": float(gross[high].mean()) if high.any() else None,
        "highconf_mean_signed_net": float(net[high].mean()) if high.any() else None,
        "highconf_median_signed_net": float(np.median(net[high])) if high.any() else None,
        "highconf_positive_net_rate": float((net[high] > 0.0).mean()) if high.any() else None,
        "highconf_cluster_weighted_net": (
            float(np.dot(net[high], weights[high]) / weighted_denominator) if weighted_denominator > 0.0 else None
        ),
        "confidence_correctness_auc": safe_auc(correct.astype(int), np.asarray(score, float)),
        "score_min": float(np.min(score)),
        "score_max": float(np.max(score)),
    }


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "highconf_mean_signed_net": float(candidate["highconf_mean_signed_net"] - baseline["highconf_mean_signed_net"]),
        "highconf_accuracy": float(candidate["highconf_accuracy"] - baseline["highconf_accuracy"]),
        "confidence_correctness_auc": float(
            candidate["confidence_correctness_auc"] - baseline["confidence_correctness_auc"]
        ),
    }


def choose_on_inner(inner_valid: pd.DataFrame) -> dict[str, Any]:
    require(len(inner_valid) >= 100, "inner validation is too small")
    baseline_score, baseline_high = apply_policy(inner_valid, 0.0)
    baseline = policy_metrics(inner_valid, baseline_score, baseline_high)
    require(baseline["highconf_n"] > 0, "inner validation has no V58 high-confidence rows")
    trials: list[dict[str, Any]] = []
    for bonus in BONUS_GRID:
        score, high = apply_policy(inner_valid, bonus)
        metrics = policy_metrics(inner_valid, score, high)
        delta = metric_delta(metrics, baseline)
        eligible = bool(
            delta["highconf_accuracy"] >= HC_ACCURACY_REGRESSION_FLOOR
            and delta["confidence_correctness_auc"] >= CONFIDENCE_AUC_REGRESSION_FLOOR
            and metrics["highconf_n"] == baseline["highconf_n"]
        )
        trials.append(
            {
                "candidate": f"AGREEMENT_BONUS_{bonus:.2f}",
                "agreement_bonus": bonus,
                "disagreement_penalty_ranking_equivalent": bonus,
                "metrics": metrics,
                "delta_vs_noop": delta,
                "accuracy_auc_guard_eligible": eligible,
            }
        )
    eligible_trials = [trial for trial in trials if trial["accuracy_auc_guard_eligible"]]
    require(eligible_trials, "no eligible agreement candidate, including no-op")
    selected = max(
        eligible_trials,
        key=lambda trial: (trial["delta_vs_noop"]["highconf_mean_signed_net"], -trial["agreement_bonus"]),
    )
    return {
        "selection_rule": (
            "maximize inner-past HC mean signed net among candidates with HC accuracy delta >= -0.005, "
            "confidence correctness AUC delta >= -0.005, and exact baseline HC count; tie -> smaller bonus"
        ),
        "baseline": baseline,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
    }


def chronology_record(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) > 0 and len(valid) > 0, f"{label} has an empty side")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label} violates strict 35-minute embargo")
    overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    require(not overlap, f"{label} has event-group overlap")
    return {
        "train_n": len(train),
        "valid_n": len(valid),
        "train_end": train_end.isoformat(),
        "valid_start": valid_start.isoformat(),
        "strict_35m_embargo_verified": True,
        "event_group_overlap": 0,
    }


def inner_partition(outer_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = outer_train.sort_values(["event_time_utc", "event_id"], kind="stable")
    boundary = max(1, min(len(ordered) - 1, int(math.floor(len(ordered) * (1.0 - INNER_VALID_FRACTION)))))
    inner_valid = ordered.iloc[boundary:].copy()
    valid_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = ordered.loc[ordered.event_time_utc < valid_start - EMBARGO].copy()
    audit = chronology_record(inner_train, inner_valid, "inner split")
    audit["inner_valid_fraction_target"] = INNER_VALID_FRACTION
    audit["inner_gap_rows"] = len(ordered) - len(inner_train) - len(inner_valid)
    return inner_train, inner_valid, audit


def run_nested_selector(sec: pd.DataFrame, smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    folds = sorted(int(value) for value in sec.fold.unique())
    require(folds == [1, 2, 3, 4], f"unexpected V58 folds: {folds}")
    audits: list[dict[str, Any]] = []
    evidence_parts: list[pd.DataFrame] = []
    first_block = sec.loc[sec.fold.eq(folds[0])].sort_values(["event_time_utc", "event_id"], kind="stable")
    first_boundary = int(math.floor(len(first_block) * FIRST_OUTER_BURN_IN_FRACTION))
    require(0 < first_boundary < len(first_block), "invalid first outer burn-in boundary")
    first_valid_start = pd.Timestamp(first_block.iloc[first_boundary].event_time_utc)

    for outer_fold in folds:
        block = sec.loc[sec.fold.eq(outer_fold)].sort_values(["event_time_utc", "event_id"], kind="stable")
        outer_valid = block.loc[block.event_time_utc >= first_valid_start].copy() if outer_fold == folds[0] else block.copy()
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = sec.loc[sec.event_time_utc < outer_start - EMBARGO].copy()
        outer_audit = chronology_record(outer_train, outer_valid, f"outer fold {outer_fold}")
        inner_train, inner_valid, inner_audit = inner_partition(outer_train)
        policy = choose_on_inner(inner_valid)
        bonus = float(policy["selected"]["agreement_bonus"])

        # Policy is locked above.  Outer outcomes are evaluated only below.
        baseline_score, baseline_high = apply_policy(outer_valid, 0.0)
        candidate_score_values, candidate_high = apply_policy(outer_valid, bonus)
        baseline_metrics = policy_metrics(outer_valid, baseline_score, baseline_high)
        candidate_metrics = policy_metrics(outer_valid, candidate_score_values, candidate_high)
        outer_delta = metric_delta(candidate_metrics, baseline_metrics)

        positions = outer_valid.index.to_numpy(int)
        sec.loc[positions, "selector_confidence_candidate"] = candidate_score_values
        sec.loc[positions, "selector_high_candidate"] = candidate_high
        sec.loc[positions, "selector_bonus"] = bonus
        sec.loc[positions, "selector_outer_fold"] = outer_fold
        sec.loc[positions, "selector_evidence_eligible"] = True
        selected_outer = sec.loc[positions].copy()
        evidence_parts.append(selected_outer)
        audits.append(
            {
                "outer_fold": outer_fold,
                "outer_chronology": outer_audit,
                "inner_chronology": inner_audit,
                "inner_train_labels_used_for_selection": False,
                "inner_valid_labels_used_for_selection": True,
                "outer_labels_used_for_selection": False,
                "policy_locked_before_outer_evaluation": True,
                "policy": policy,
                "outer_baseline": baseline_metrics,
                "outer_candidate": candidate_metrics,
                "outer_delta": outer_delta,
                "outer_hc_count_exactly_preserved": (
                    candidate_metrics["highconf_n"] == baseline_metrics["highconf_n"]
                ),
            }
        )
        if smoke:
            break

    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "selector outer evidence repeats an event")
    return sec, evidence, audits


def aggregate_metrics(frame: pd.DataFrame, candidate: bool) -> dict[str, Any]:
    if candidate:
        score = frame.selector_confidence_candidate.to_numpy(float)
        high = frame.selector_high_candidate.to_numpy(bool)
    else:
        score = frame.confidence_signal.to_numpy(float)
        high = frame.high_conf.to_numpy(bool)
    return policy_metrics(frame, score, high)


def bootstrap_policy_delta(frame: pd.DataFrame, draws: int) -> dict[str, Any]:
    require(draws > 0, "bootstrap draw count must be positive")
    work = frame.copy()
    work["candidate_selected_net"] = np.where(work.selector_high_candidate, work.signed_net, 0.0)
    work["baseline_selected_net"] = np.where(work.high_conf, work.signed_net, 0.0)
    work["candidate_selected_correct"] = np.where(work.selector_high_candidate, work.direction_correct, 0.0)
    work["baseline_selected_correct"] = np.where(work.high_conf, work.direction_correct, 0.0)
    work["candidate_selected_n"] = work.selector_high_candidate.astype(int)
    work["baseline_selected_n"] = work.high_conf.astype(int)
    grouped = (
        work.groupby(["selector_outer_fold", "event_group_id"], sort=True)
        .agg(
            candidate_net=("candidate_selected_net", "sum"),
            baseline_net=("baseline_selected_net", "sum"),
            candidate_correct=("candidate_selected_correct", "sum"),
            baseline_correct=("baseline_selected_correct", "sum"),
            candidate_n=("candidate_selected_n", "sum"),
            baseline_n=("baseline_selected_n", "sum"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(SEED)
    net_delta = np.empty(draws, dtype=float)
    accuracy_delta = np.empty(draws, dtype=float)
    by_fold = [part.reset_index(drop=True) for _, part in grouped.groupby("selector_outer_fold", sort=True)]
    for draw in range(draws):
        totals = np.zeros(6, dtype=float)
        for part in by_fold:
            sampled = part.iloc[rng.integers(0, len(part), size=len(part))]
            totals += sampled[
                ["candidate_net", "baseline_net", "candidate_correct", "baseline_correct", "candidate_n", "baseline_n"]
            ].sum().to_numpy(float)
        require(totals[4] > 0 and totals[5] > 0, "bootstrap draw has zero selected rows")
        net_delta[draw] = totals[0] / totals[4] - totals[1] / totals[5]
        accuracy_delta[draw] = totals[2] / totals[4] - totals[3] / totals[5]
    quantiles = (0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975)
    return {
        "method": "paired stratified outer-fold event_group cluster bootstrap over fixed policies",
        "seed": SEED,
        "draws": draws,
        "event_groups": int(work.event_group_id.nunique()),
        "event_group_repeated_rows": int(len(work) - work.event_group_id.nunique()),
        "net_delta_quantiles": {str(q): float(np.quantile(net_delta, q)) for q in quantiles},
        "net_delta_mean": float(net_delta.mean()),
        "net_delta_probability_gt_zero": float((net_delta > 0.0).mean()),
        "accuracy_delta_quantiles": {str(q): float(np.quantile(accuracy_delta, q)) for q in quantiles},
        "accuracy_delta_mean": float(accuracy_delta.mean()),
        "accuracy_delta_probability_ge_minus_0_005": float(
            (accuracy_delta >= HC_ACCURACY_REGRESSION_FLOOR).mean()
        ),
    }


def risk_coverage(frame: pd.DataFrame, name: str, score: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    correct = frame.direction_correct.to_numpy(bool)
    net = frame.signed_net.to_numpy(float)
    event_id = frame.event_id.astype(str).to_numpy()
    for coverage in (0.10, 0.15, 0.20, 0.25, 0.30):
        k = max(1, int(round(len(frame) * coverage)))
        selected = stable_top_k(frame, score, k)
        rows.append(
            {
                "policy": name,
                "target_coverage": coverage,
                "n": k,
                "actual_coverage": float(selected.mean()),
                "accuracy": float(correct[selected].mean()),
                "mean_signed_net": float(net[selected].mean()),
                "median_signed_net": float(np.median(net[selected])),
                "event_id_tiebreak_sha256": hashlib.sha256("\n".join(sorted(event_id[selected])).encode()).hexdigest(),
            }
        )
    return rows


def tail_economics(frame: pd.DataFrame, high_column: str) -> dict[str, Any]:
    selected = frame.loc[frame[high_column].astype(bool), "signed_net"].to_numpy(float)
    require(len(selected) > 10, "too few high-confidence rows for tail audit")
    lower, upper = np.quantile(selected, [0.01, 0.99])
    winsorized = np.clip(selected, lower, upper)
    top_removed = np.delete(selected, np.argsort(selected)[-min(5, len(selected) - 1):])
    return {
        "n": len(selected),
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "p01": float(lower),
        "p05": float(np.quantile(selected, 0.05)),
        "p95": float(np.quantile(selected, 0.95)),
        "p99": float(upper),
        "winsorized_1pct_mean": float(winsorized.mean()),
        "top_5_positive_removed_mean": float(top_removed.mean()),
        "worst_5_mean": float(np.sort(selected)[:5].mean()),
    }


def candidate_full_frame(v58: pd.DataFrame, sec: pd.DataFrame, material_pass: bool) -> pd.DataFrame:
    final = v58.copy()
    original_prob = final.prob.to_numpy(float).copy()
    original_confidence = final.confidence_signal.to_numpy(float).copy()
    original_high = final.high_conf.to_numpy(bool).copy()
    if material_pass:
        lookup = sec.set_index("event_id")
        mask = final.source_family.eq("US_SEC")
        final.loc[mask, "confidence_signal"] = final.loc[mask, "event_id"].map(
            lookup.selector_confidence_candidate
        ).to_numpy(float)
        final.loc[mask, "high_conf"] = final.loc[mask, "event_id"].map(
            lookup.selector_high_candidate
        ).to_numpy(bool)
        final.loc[mask, "family"] = "V69_US_SEC_STRUCTURED_AGREEMENT_SELECTOR_V1"
    require(np.array_equal(final.prob.to_numpy(float), original_prob), "V69 changed V58 direction probability")
    require(
        np.array_equal(final.prob.to_numpy(float) >= 0.5, original_prob >= 0.5),
        "V69 changed V58 direction prediction",
    )
    if not material_pass:
        require(np.array_equal(final.confidence_signal.to_numpy(float), original_confidence), "V69 fallback confidence differs")
        require(np.array_equal(final.high_conf.to_numpy(bool), original_high), "V69 fallback high_conf differs")
    return final


def evaluate(
    v58: pd.DataFrame,
    sec: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    bootstrap_draws: int,
) -> dict[str, Any]:
    baseline_outer = aggregate_metrics(evidence, candidate=False)
    candidate_outer = aggregate_metrics(evidence, candidate=True)
    outer_delta = metric_delta(candidate_outer, baseline_outer)
    bootstrap = bootstrap_policy_delta(evidence, bootstrap_draws)
    fold_deltas = [float(audit["outer_delta"]["highconf_mean_signed_net"]) for audit in audits]
    positive_fraction = float(np.mean(np.asarray(fold_deltas) > 0.0))
    nonnegative_fraction = float(np.mean(np.asarray(fold_deltas) >= -1e-15))

    candidate_diagnostic = candidate_full_frame(v58, sec, material_pass=True)
    baseline_summary = v44.summarize(v58)
    candidate_summary = v44.summarize(candidate_diagnostic)
    base_sec_hc = int(v58.loc[v58.source_family.eq("US_SEC"), "high_conf"].sum())
    candidate_sec_hc = int(candidate_diagnostic.loc[candidate_diagnostic.source_family.eq("US_SEC"), "high_conf"].sum())
    baseline_total_hc = int(v58.high_conf.sum())
    candidate_total_hc = int(candidate_diagnostic.high_conf.sum())
    coverage = float(candidate_diagnostic.high_conf.mean())
    full_base_net = float(
        baseline_summary["metrics"]["by_source_family"]["US_SEC"]["strategy_mean_signed_net"]
    )
    full_candidate_net = float(
        candidate_summary["metrics"]["by_source_family"]["US_SEC"]["strategy_mean_signed_net"]
    )
    probability_hash = array_sha256(v58.prob.to_numpy(np.float64))
    candidate_probability_hash = array_sha256(candidate_diagnostic.prob.to_numpy(np.float64))
    direction_hash = array_sha256((v58.prob.to_numpy(float) >= 0.5).astype(np.uint8))
    candidate_direction_hash = array_sha256((candidate_diagnostic.prob.to_numpy(float) >= 0.5).astype(np.uint8))
    checks = {
        "outer_hc_economic_net_delta_gt_0": outer_delta["highconf_mean_signed_net"] > 0.0,
        "full_us_sec_hc_economic_net_delta_gt_0": full_candidate_net > full_base_net,
        "bootstrap_net_delta_probability_gt_zero_ge_0_85": (
            bootstrap["net_delta_probability_gt_zero"] >= BOOTSTRAP_POSITIVE_PROBABILITY_FLOOR
        ),
        "bootstrap_net_delta_median_gt_0": bootstrap["net_delta_quantiles"]["0.5"] > 0.0,
        "positive_fold_fraction_ge_0_75": positive_fraction >= POSITIVE_FOLD_FRACTION_FLOOR,
        "all_outer_folds_net_nonnegative": nonnegative_fraction == 1.0,
        "outer_hc_accuracy_delta_ge_minus_0_005": (
            outer_delta["highconf_accuracy"] >= HC_ACCURACY_REGRESSION_FLOOR
        ),
        "outer_confidence_correctness_auc_delta_ge_minus_0_005": (
            outer_delta["confidence_correctness_auc"] >= CONFIDENCE_AUC_REGRESSION_FLOOR
        ),
        "us_sec_hc_count_exactly_preserved": candidate_sec_hc == base_sec_hc,
        "overall_hc_count_exactly_preserved": candidate_total_hc == baseline_total_hc,
        "overall_coverage_ge_0_15": coverage >= OVERALL_COVERAGE_FLOOR,
        "v58_probability_sha256_exact": probability_hash == candidate_probability_hash,
        "v58_direction_sha256_exact": direction_hash == candidate_direction_hash,
        "raw_feature_sign_flip_not_performed": True,
    }
    material_pass = bool(all(checks.values()))
    final = candidate_full_frame(v58, sec, material_pass=material_pass)
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_NOOP",
        "material_pass": material_pass,
        "material_checks": checks,
        "predeclared_thresholds": {
            "hc_accuracy_regression_floor": HC_ACCURACY_REGRESSION_FLOOR,
            "confidence_correctness_auc_regression_floor": CONFIDENCE_AUC_REGRESSION_FLOOR,
            "bootstrap_positive_probability_floor": BOOTSTRAP_POSITIVE_PROBABILITY_FLOOR,
            "positive_fold_fraction_floor": POSITIVE_FOLD_FRACTION_FLOOR,
            "overall_coverage_floor": OVERALL_COVERAGE_FLOOR,
            "transaction_cost": COST,
        },
        "outer_baseline": baseline_outer,
        "outer_candidate": candidate_outer,
        "outer_delta": outer_delta,
        "outer_evidence_rows": len(evidence),
        "outer_evidence_fraction_of_us_sec": float(len(evidence) / len(sec)),
        "outer_fold_net_deltas": fold_deltas,
        "positive_fold_fraction": positive_fraction,
        "nonnegative_fold_fraction": nonnegative_fraction,
        "bootstrap": bootstrap,
        "full_baseline_summary": baseline_summary,
        "full_candidate_diagnostic_summary": candidate_summary,
        "full_selected_summary": v44.summarize(final),
        "full_us_sec_hc_net": {
            "baseline": full_base_net,
            "candidate": full_candidate_net,
            "delta": full_candidate_net - full_base_net,
        },
        "hc_count_coverage": {
            "baseline_us_sec_hc_n": base_sec_hc,
            "candidate_us_sec_hc_n": candidate_sec_hc,
            "baseline_total_hc_n": baseline_total_hc,
            "candidate_total_hc_n": candidate_total_hc,
            "baseline_overall_coverage": float(v58.high_conf.mean()),
            "candidate_overall_coverage": coverage,
        },
        "direction_immutability": {
            "baseline_probability_sha256": probability_hash,
            "candidate_probability_sha256": candidate_probability_hash,
            "baseline_direction_sha256": direction_hash,
            "candidate_direction_sha256": candidate_direction_hash,
            "max_abs_probability_delta": float(
                np.max(np.abs(v58.prob.to_numpy(float) - candidate_diagnostic.prob.to_numpy(float)))
            ),
            "direction_mismatch_n": int(
                np.sum((v58.prob.to_numpy(float) >= 0.5) != (candidate_diagnostic.prob.to_numpy(float) >= 0.5))
            ),
            "exact_unchanged": probability_hash == candidate_probability_hash and direction_hash == candidate_direction_hash,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact V58 prob/confidence_signal/high_conf" if not material_pass else None,
            "exact_arrays_verified": (
                np.array_equal(final.prob.to_numpy(float), v58.prob.to_numpy(float))
                and (
                    material_pass
                    or (
                        np.array_equal(final.confidence_signal.to_numpy(float), v58.confidence_signal.to_numpy(float))
                        and np.array_equal(final.high_conf.to_numpy(bool), v58.high_conf.to_numpy(bool))
                    )
                )
            ),
        },
        "final_frame": final,
    }


def reports(
    v58: pd.DataFrame,
    sec: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    status = evaluation["status"]
    material_pass = evaluation["material_pass"]
    final = evaluation["final_frame"]
    validation = (
        "V58 probability/direction frozen; V59 structured direction agreement used only in bounded confidence ranking; "
        "predeclared bonus grid; nested chronological inner-past selection; strict 35-minute embargo; outer outcomes never "
        "used for policy selection; exact per-block US_SEC HC count; fail-closed exact V58 fallback."
    )
    selected_contract = dict(evaluation["full_selected_summary"])
    # Hypothesis-specific material gates select the V69 policy versus the
    # exact V58 fallback.  They must remain separate from the canonical
    # program-level research/seal gates produced by v44.summarize().
    selected_contract["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": material_pass,
    }
    compact_evaluation = {key: value for key, value in evaluation.items() if key != "final_frame"}
    comparison = {
        "version": "V69",
        "hypothesis": HYPOTHESIS,
        "status": status,
        "material_pass": material_pass,
        "selected_policy": "V69_US_SEC_AGREEMENT_SELECTOR" if material_pass else "V58_EXACT_NOOP",
        "models": {
            "V58_BASELINE": evaluation["full_baseline_summary"],
            "V69_CANDIDATE_DIAGNOSTIC": evaluation["full_candidate_diagnostic_summary"],
            "V69_FAIL_CLOSED_SELECTED": evaluation["full_selected_summary"],
        },
        "evaluation": compact_evaluation,
        "input_audit": input_audit,
        "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V69",
        "hypothesis": HYPOTHESIS,
        "status": status,
        "opened_dev_only": True,
        "seal_outcomes_loaded": False,
        "dev_extension_loaded": False,
        "selected": selected_contract,
        "candidate": evaluation["full_candidate_diagnostic_summary"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_v58_noop": not material_pass,
        "seal_authorized": False,
    }
    transfer = {
        "version": "V69",
        "hypothesis": HYPOTHESIS,
        "status": status,
        "seal_outcomes_loaded": False,
        "selected_oof_by_source": evaluation["full_selected_summary"]["metrics"]["by_source_family"],
        "selected_oof_by_market": evaluation["full_selected_summary"]["metrics"]["by_market"],
        "scope": "US_SEC confidence/high_conf only",
        "all_non_us_sec_rows_unchanged": True,
        "direction_exact_unchanged": evaluation["direction_immutability"]["exact_unchanged"],
        "method": validation,
    }
    selector_report = {
        "version": "V69",
        "hypothesis": HYPOTHESIS,
        "status": status,
        "candidate_contract": {
            "bonus_grid": list(BONUS_GRID),
            "formula": "confidence=(V58_confidence + bonus*I[V58_direction==V59_structured_direction])/(1+bonus)",
            "disagreement_penalty_equivalence": (
                "base + bonus*agreement and base - bonus*disagreement differ only by a constant, so rankings are identical"
            ),
            "v58_direction_probability_mutable": False,
            "v59_probability_output_authorized": False,
            "ranking_uses_outcome": False,
            "raw_sign_flip": False,
        },
        "nested_selection_audits": audits,
        "evaluation": compact_evaluation,
        "input_audit": input_audit,
        "validation": validation,
    }
    selection_report = {
        "version": "V69",
        "hypothesis": HYPOTHESIS,
        "status": status,
        "selection_space": {
            "bonus_grid": list(BONUS_GRID),
            "first_outer_burn_in_fraction": FIRST_OUTER_BURN_IN_FRACTION,
            "inner_valid_fraction": INNER_VALID_FRACTION,
            "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        },
        "folds": audits,
        "outer_labels_used_for_selection": False,
        "material_checks": evaluation["material_checks"],
    }
    agreement_audit = {}
    for agrees, part in evidence.groupby("structured_agreement"):
        agreement_audit["AGREE" if agrees else "DISAGREE"] = {
            "n": len(part),
            "direction_accuracy": float(part.direction_correct.mean()),
            "mean_signed_net": float(part.signed_net.mean()),
            "baseline_highconf_n": int(part.high_conf.sum()),
            "candidate_highconf_n": int(part.selector_high_candidate.sum()),
        }
    economic = {
        "version": "V69",
        "hypothesis": HYPOTHESIS,
        "status": status,
        "transaction_cost": COST,
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_delta": evaluation["outer_delta"],
        "full_us_sec_hc_net": evaluation["full_us_sec_hc_net"],
        "fold_net_deltas": evaluation["outer_fold_net_deltas"],
        "positive_fold_fraction": evaluation["positive_fold_fraction"],
        "bootstrap": evaluation["bootstrap"],
        "tail_baseline_outer": tail_economics(evidence, "high_conf"),
        "tail_candidate_outer": tail_economics(evidence, "selector_high_candidate"),
    }
    fold_rows = []
    for audit in audits:
        selected = audit["policy"]["selected"]
        fold_rows.append(
            {
                "outer_fold": audit["outer_fold"],
                "outer_train_n": audit["outer_chronology"]["train_n"],
                "inner_train_n": audit["inner_chronology"]["train_n"],
                "inner_valid_n": audit["inner_chronology"]["valid_n"],
                "outer_valid_n": audit["outer_chronology"]["valid_n"],
                "outer_train_end": audit["outer_chronology"]["train_end"],
                "outer_valid_start": audit["outer_chronology"]["valid_start"],
                "selected_bonus": selected["agreement_bonus"],
                "inner_selected_net_delta": selected["delta_vs_noop"]["highconf_mean_signed_net"],
                "outer_net_delta": audit["outer_delta"]["highconf_mean_signed_net"],
                "outer_accuracy_delta": audit["outer_delta"]["highconf_accuracy"],
                "outer_confidence_auc_delta": audit["outer_delta"]["confidence_correctness_auc"],
                "baseline_hc_n": audit["outer_baseline"]["highconf_n"],
                "candidate_hc_n": audit["outer_candidate"]["highconf_n"],
                "strict_35m_embargo": True,
                "outer_labels_used_for_selection": False,
                "raw_feature_sign_flip_performed": False,
            }
        )
    risk_rows = risk_coverage(
        evidence, "V58_BASELINE", evidence.confidence_signal.to_numpy(float)
    ) + risk_coverage(
        evidence, "V69_AGREEMENT_SELECTOR", evidence.selector_confidence_candidate.to_numpy(float)
    )
    direction_audit = {
        "version": "V69",
        "hypothesis": HYPOTHESIS,
        **evaluation["direction_immutability"],
        "v59_probability_used_for_direction_output": False,
        "raw_feature_sign_flip_performed": False,
    }
    oof_columns = [
        "event_id",
        "event_group_id",
        "event_time_utc",
        "ticker",
        "source_family",
        "fold",
        "y",
        "fwd_ret_30m",
        "prob",
        "confidence_signal",
        "high_conf",
        "structured_prob",
        "v58_direction",
        "v59_structured_direction",
        "structured_agreement",
        "selector_confidence_candidate",
        "selector_high_candidate",
        "selector_bonus",
        "selector_outer_fold",
        "selector_evidence_eligible",
        "direction_correct",
        "signed_gross",
        "signed_net",
    ]
    selected_columns = list(v58.columns)
    json_reports = {
        "MODEL_COMPARISON.json": comparison,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": transfer,
        "US_SEC_STRUCTURED_AGREEMENT_SELECTOR_REPORT.json": selector_report,
        "AGREEMENT_SELECTOR_SELECTION_REPORT.json": selection_report,
        "ECONOMIC_ROBUSTNESS_REPORT.json": economic,
        "BOOTSTRAP_DELTA_REPORT.json": evaluation["bootstrap"],
        "DIRECTION_IMMUTABILITY_AUDIT.json": direction_audit,
        "AGREEMENT_COHORT_AUDIT.json": agreement_audit,
    }
    table_reports = {
        "US_SEC_AGREEMENT_FOLD_AUDIT.csv": pd.DataFrame(fold_rows),
        "RISK_COVERAGE_REPORT.csv": pd.DataFrame(risk_rows),
        "US_SEC_AGREEMENT_SELECTOR_OOF.csv.gz": sec[oof_columns].copy(),
        "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz": final[selected_columns].copy(),
    }
    return json_reports, table_reports


def write_with_controller(
    json_reports: dict[str, Any],
    table_reports: dict[str, pd.DataFrame],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in json_reports.items():
        atomic_json(payload, OUT / name)
    for name, frame in table_reports.items():
        atomic_csv(frame, OUT / name)
    experiment_id = hashlib.sha256(
        (HYPOTHESIS + "|" + "|".join(input_audit["input_sha256"].values())).encode("utf-8")
    ).hexdigest()
    run_status = {
        "version": VERSION,
        "status": evaluation["status"],
        "phase": "ROBUST_SURVIVOR" if evaluation["material_pass"] else "RESEARCH_FAIL",
        "exit_code": 0,
        "completed_at": now(),
        "experiment_id": experiment_id,
        "seal_state": "UNOPENED",
        "final_status": "CONTINUE",
        "output_scope": str(OUT.relative_to(ROOT)) if OUT.is_relative_to(ROOT) else str(OUT),
    }
    atomic_json(run_status, OUT / "RUN_STATUS.json")
    files: dict[str, Any] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}:
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "experiment_id": experiment_id,
        "files": files,
    }
    atomic_json(manifest, OUT / "ARTIFACT_MANIFEST.json")
    commit = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "committed_at": now(),
        "experiment_id": experiment_id,
        "input_sha256": input_audit["input_sha256"],
        "material_fingerprint": hashlib.sha256(
            json.dumps(
                {
                    "hypothesis": HYPOTHESIS,
                    "bonus_grid": BONUS_GRID,
                    "material_thresholds": evaluation["predeclared_thresholds"],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
    }
    atomic_json(commit, OUT / "COMMIT.json")
    return {
        "run_status_sha256": sha256(OUT / "RUN_STATUS.json"),
        "artifact_manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
        "commit_sha256": sha256(OUT / "COMMIT.json"),
        "artifact_count": len(files) + 2,
        "experiment_id": experiment_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    v58, v59, input_audit = verify_and_load()
    sec = add_agreement(v58, v59)
    if args.audit_only:
        print(
            json.dumps(
                json_clean(
                    {
                        "status": "AUDIT_OK",
                        "hypothesis": HYPOTHESIS,
                        "input_audit": input_audit,
                        "agreement_rate": float(sec.structured_agreement.mean()),
                        "output_written": False,
                    }
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    sec, evidence, audits = run_nested_selector(sec, smoke=args.smoke_test)
    evaluation = evaluate(
        v58,
        sec,
        evidence,
        audits,
        bootstrap_draws=250 if args.smoke_test else BOOTSTRAP_DRAWS,
    )
    summary = {
        "status": evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "material_pass": evaluation["material_pass"],
        "selected_bonuses": [audit["policy"]["selected"]["agreement_bonus"] for audit in audits],
        "outer_hc_net_delta": evaluation["outer_delta"]["highconf_mean_signed_net"],
        "outer_hc_accuracy_delta": evaluation["outer_delta"]["highconf_accuracy"],
        "confidence_correctness_auc_delta": evaluation["outer_delta"]["confidence_correctness_auc"],
        "bootstrap_probability_delta_gt_zero": evaluation["bootstrap"]["net_delta_probability_gt_zero"],
        "positive_fold_fraction": evaluation["positive_fold_fraction"],
        "direction_exact_unchanged": evaluation["direction_immutability"]["exact_unchanged"],
        "hc_count_coverage": evaluation["hc_count_coverage"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_v58_noop": not evaluation["material_pass"],
    }
    if args.smoke_test:
        print(json.dumps(json_clean({**summary, "status": "SMOKE_OK", "output_written": False}), indent=2))
        return
    json_reports, table_reports = reports(v58, sec, evidence, audits, evaluation, input_audit)
    output_names = sorted([*json_reports, *table_reports, "RUN_STATUS.json", "ARTIFACT_MANIFEST.json", "COMMIT.json"])
    if args.dry_run:
        print(
            json.dumps(
                json_clean({**summary, "status": "DRY_RUN_OK", "would_write": output_names, "output_written": False}),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    controller = write_with_controller(json_reports, table_reports, evaluation, input_audit)
    print(
        json.dumps(
            json_clean(
                {
                    **summary,
                    "output_dir": str(OUT),
                    "files": output_names,
                    "controller": controller,
                    "seal_authorized": False,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
