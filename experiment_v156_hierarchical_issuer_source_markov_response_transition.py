"""V156 hierarchical issuer/source Markov response-transition direction.

The initially proposed static issuer/source UP-rate and mean-return hierarchy
was rejected before implementation because V40, V61, V71 and V73 already fit
past-only marginal response priors.  V156 instead models new sequential
information: on equal-weight event groups inside every strict-past same-market
train cut, it counts first-order transitions from each entity's immediately
previous observed direction to its next direction.  Fixed Jeffreys market
transition posteriors shrink source-family transition tables, which in turn
shrink issuer/ticker transition tables.  A query uses the last response state
observed before the validation boundary; an unseen issuer falls back to its
source-family state/table and then market state/table.

No current/outer outcome is used, no numeric state or text is synthesized, and
there is no fitted global coefficient head.  Inner-past evidence selects exact
V69 or fixed .25/.50 blends with the transition probability.  Outer labels are
evaluation-only after a strict 35-minute embargo and event-group purge.  V69
confidence/high-confidence remain exact, and any current material-gate failure
restores the exact entire atomic V69 DataFrame.

Audit, US:2 smoke and KR:2 support use CPU30-31/two threads/no GPU and write
nothing.  Full execution is accepted only through no-argument
MARKET_BIO_VERSION_OUTPUT binding.
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
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import experiment_v150_multichannel_causal_temporal_channel_mlp_mixer as base


ROOT = base.ROOT
V69_DIR = base.V69_DIR
DEFAULT_OUT = ROOT / "research" / "local_run_outputs" / "V156_HIERARCHICAL_MARKOV_RESPONSE_TRANSITION_V1"
VERSION = 156
HYPOTHESIS = "HIERARCHICAL_ISSUER_SOURCE_MARKOV_RESPONSE_TRANSITION_DIRECTION_V1"
EXPECTED_MARKET_FOLD_ROWS = base.EXPECTED_MARKET_FOLD_ROWS
SCAFFOLD_PATH = ROOT / "experiment_v150_multichannel_causal_temporal_channel_mlp_mixer.py"
EXPECTED_SCAFFOLD_SHA256 = "b4b0fc2789f27e5556ba1ec18936bafdff93779fea1f31aeee976852a2f30393"

MARKET_JEFFREYS = 0.5
SOURCE_STRENGTH = 12.0
ISSUER_STRENGTH = 8.0
SEED = 15601
BOOTSTRAP_DRAWS = 2000
SMOKE_BOOTSTRAP_DRAWS = 250
SMOKE_THREADS = 2
ARCHITECTURES = (
    {"name": "HIERARCHICAL_MARKOV_RESPONSE_W0.25", "weight": 0.25},
    {"name": "HIERARCHICAL_MARKOV_RESPONSE_W0.50", "weight": 0.50},
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

STATIC_SPEC = {
    "training_unit": "equal event group",
    "sequence_order": "event_time_utc then event_id",
    "response_state": "immediately previous observed binary direction",
    "transition_target": "next binary direction",
    "market_prior": "two-state first-order transition table with Jeffreys Beta(0.5,0.5)",
    "source_prior": f"source-family transition table shrunk to market with fixed strength {SOURCE_STRENGTH:g}",
    "issuer_prior": f"ticker transition table shrunk to target source-family with fixed strength {ISSUER_STRENGTH:g}",
    "unseen_issuer_fallback": "source-family last state/table, then market last state/table",
    "current_or_target_response_update": False,
    "static_marginal_up_rate_prior": False,
    "mean_signed_return_prior": False,
    "global_discriminative_head": False,
    "numeric_text_event_taxonomy_features": False,
}


def verify_authority() -> dict[str, Any]:
    require(sha256(SCAFFOLD_PATH) == EXPECTED_SCAFFOLD_SHA256, "pinned V150 utility scaffold changed")
    audit = base.verify_authority()
    audit["code_dependency"] = {
        "path": SCAFFOLD_PATH.name,
        "sha256": EXPECTED_SCAFFOLD_SHA256,
        "purpose": "immutable authority, chronology, metrics, canonical gate, blending, nested-loop and atomic IO utilities only; no V150 output is read",
    }
    audit["v156_access_contract"] = {
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


def equal_event_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = frame.sort_values(["event_time_utc", "event_id"], kind="stable").copy()
    consistency = ordered.groupby("event_group_id", sort=False).agg(
        y_n=("y", "nunique"), ticker_n=("ticker", "nunique"),
        source_n=("source_family", "nunique"), market_n=("market", "nunique"),
    )
    require(
        int(consistency.to_numpy().max()) == 1,
        "V156 event-group response/entity/source consistency failed",
    )
    groups = ordered.drop_duplicates("event_group_id", keep="first").copy()
    groups["ticker"] = groups.ticker.fillna("UNKNOWN").astype(str)
    groups["source_family"] = groups.source_family.fillna("UNKNOWN").astype(str)
    groups = groups.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(groups.y.isin([0, 1]).all(), "V156 group target invalid")
    return groups, {
        "row_n": len(frame),
        "event_group_n": len(groups),
        "duplicate_rows_removed": len(frame) - len(groups),
        "equal_event_group_weight": True,
        "group_event_id_sha256": array_sha256(groups.event_id.astype(str).to_numpy(dtype="U")),
    }


def transition_counts(
    groups: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> tuple[dict[tuple[str, ...], np.ndarray], dict[tuple[str, ...], int]]:
    tables: dict[tuple[str, ...], np.ndarray] = {}
    last_state: dict[tuple[str, ...], int] = {}
    for row in groups.itertuples(index=False):
        key = tuple(str(getattr(row, column)) for column in key_columns)
        current = int(row.y)
        if key in last_state:
            table = tables.setdefault(key, np.zeros((2, 2), dtype=float))
            table[last_state[key], current] += 1.0
        last_state[key] = current
    return tables, last_state


def market_probability(table: np.ndarray, previous: int) -> float:
    return float(
        (table[previous, 1] + MARKET_JEFFREYS)
        / (table[previous].sum() + 2.0 * MARKET_JEFFREYS)
    )


def shrunk_probability(table: np.ndarray | None, previous: int, parent: float, strength: float) -> tuple[float, float]:
    if table is None:
        return float(parent), 0.0
    support = float(table[previous].sum())
    probability = float((table[previous, 1] + strength * parent) / (support + strength))
    return probability, support


def markov_direction(train: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    require(train.market.nunique() == 1 and target_frame.market.nunique() == 1, "V156 expects one market per fit")
    groups, group_audit = equal_event_groups(train)
    market_tables, market_last = transition_counts(groups, ("market",))
    source_tables, source_last = transition_counts(groups, ("source_family",))
    issuer_tables, issuer_last = transition_counts(groups, ("ticker",))
    market_key = (str(groups.market.iloc[0]),)
    market_table = market_tables.get(market_key, np.zeros((2, 2), dtype=float))
    market_state = market_last[market_key]
    require(market_table.sum() >= 40.0, "V156 insufficient market transitions")

    probability = np.empty(len(target_frame), dtype=float)
    routes = {"issuer": 0, "source": 0, "market": 0}
    issuer_support: list[float] = []
    source_support: list[float] = []
    for position, row in enumerate(target_frame.itertuples(index=False)):
        source_key = (str(row.source_family),)
        issuer_key = (str(row.ticker),)
        if issuer_key in issuer_last:
            previous = issuer_last[issuer_key]
            market_parent = market_probability(market_table, previous)
            source_parent, source_n = shrunk_probability(
                source_tables.get(source_key), previous, market_parent, SOURCE_STRENGTH,
            )
            current, issuer_n = shrunk_probability(
                issuer_tables.get(issuer_key), previous, source_parent, ISSUER_STRENGTH,
            )
            probability[position] = current
            routes["issuer"] += 1
            issuer_support.append(issuer_n)
            source_support.append(source_n)
        elif source_key in source_last:
            previous = source_last[source_key]
            market_parent = market_probability(market_table, previous)
            current, source_n = shrunk_probability(
                source_tables.get(source_key), previous, market_parent, SOURCE_STRENGTH,
            )
            probability[position] = current
            routes["source"] += 1
            source_support.append(source_n)
        else:
            probability[position] = market_probability(market_table, market_state)
            routes["market"] += 1
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    require(np.isfinite(probability).all(), "V156 probability invalid")
    return probability, {
        "train_n": len(train),
        "target_n": len(target_frame),
        "causal_historical_response_only": True,
        "equal_event_groups": group_audit,
        "static_spec": STATIC_SPEC,
        "transition_tables": {
            "market_key_n": len(market_tables),
            "source_key_n": len(source_tables),
            "issuer_key_n": len(issuer_tables),
            "market_table": market_table.tolist(),
            "market_table_sha256": array_sha256(market_table),
            "market_last_state": market_state,
            "source_transition_total": float(sum(table.sum() for table in source_tables.values())),
            "issuer_transition_total": float(sum(table.sum() for table in issuer_tables.values())),
        },
        "prediction_routes": routes,
        "issuer_previous_state_coverage": float(routes["issuer"] / len(target_frame)),
        "source_or_market_fallback_coverage": float((routes["source"] + routes["market"]) / len(target_frame)),
        "issuer_condition_support_mean": float(np.mean(issuer_support)) if issuer_support else 0.0,
        "source_condition_support_mean": float(np.mean(source_support)) if source_support else 0.0,
        "target_rows_or_labels_used_to_update_transition_tables_or_last_states": False,
        "outer_labels_used": False,
        "static_marginal_response_prior": False,
        "mean_signed_return_prior": False,
        "v40_v61_v71_v73_static_response_lookup": False,
        "v130_v69_correctness_reliability_prior": False,
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
    challenger, model_audit = markov_direction(inner_train, inner_valid)
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
        "selection_rule": "inner-past OOF only; exact V69 no-op or fixed .25/.50 hierarchical Markov-response blend; fixed 2*AUC+BA with BA/net/supported-source safety",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "transition_order_hierarchy_strength_blend_or_source_floor_micro_tuning": False,
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
        base.mixer_direction = markov_direction
        base.ARCHITECTURES = ARCHITECTURES
        with contextlib.redirect_stdout(io.StringIO()):
            diagnostic, evidence, audits = base.run_nested(
                dev, champion, smoke=smoke, smoke_market=smoke_market,
            )
    finally:
        base.choose_inner = old_choose
        base.mixer_direction = old_direction
        base.ARCHITECTURES = old_architectures
    diagnostic.rename(columns={"v150_model": "v156_model"}, inplace=True)
    evidence.rename(columns={"mixer_prob": "markov_prob", "v150_model": "v156_model"}, inplace=True)
    for audit in audits:
        selected = audit["policy"]["selected"]
        print(
            f"[V156 MARKOV-RESPONSE] market={audit['market']} fold={audit['fold']} "
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
    result["method"] = "paired market-fold-time block bootstrap over inner-locked V156 hierarchical Markov-response policy"
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
    require(candidate_summary["research_gate"] == canonical_candidate, "V156 canonical mismatch")
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
    markov_contract = all(
        audit[side]["causal_historical_response_only"]
        and audit[side]["equal_event_groups"]["equal_event_group_weight"]
        and audit[side]["static_spec"]["response_state"] == "immediately previous observed binary direction"
        and not audit[side]["target_rows_or_labels_used_to_update_transition_tables_or_last_states"]
        and not audit[side]["outer_labels_used"]
        and not audit[side]["static_marginal_response_prior"]
        and not audit[side]["mean_signed_return_prior"]
        and not audit[side]["v40_v61_v71_v73_static_response_lookup"]
        and not audit[side]["v130_v69_correctness_reliability_prior"]
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
        "hierarchical_issuer_source_markov_response_contract_verified": markov_contract,
    }
    material_pass = bool(not smoke and all(checks.values()))
    selected_frame = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected gate mismatch")
    if not material_pass:
        require(selected_frame.equals(champion), "V156 fallback is not exact entire V69")
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
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "V156 nested key mismatch")
    return gate


def write_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    require(bool(CONTROLLER_OUTPUT), "V156 full output requires MARKET_BIO_VERSION_OUTPUT authority")
    require(out.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V156 output is not controller-bound")
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    gate = material_gate(evaluation)
    selected = json.loads(json.dumps(clean(evaluation["selected_summary"])))
    selected["material_gate"] = gate
    require(gate["nested_bootstrap"]["contract_key"] == "selected.material_gate.nested_bootstrap", "nested bootstrap key mismatch")
    require("bootstrap" in selected["robustness"], "selected native robustness bootstrap missing")
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V156", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {
                "V69_CHAMPION": evaluation["baseline_summary"],
                "V156_DIAGNOSTIC_HIERARCHICAL_MARKOV_RESPONSE": evaluation["candidate_summary"],
                "V156_FAIL_CLOSED_SELECTED": selected,
            },
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V156", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": selected, "candidate": evaluation["candidate_summary"],
            "material_gate": gate, "material_checks": evaluation["material_checks"],
            "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V156", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": selected["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": selected["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "material_gate": gate, "nested_bootstrap": evaluation["nested_bootstrap"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
        },
        "V156_HIERARCHICAL_ISSUER_SOURCE_MARKOV_RESPONSE_REPORT.json": {
            "version": "V156", "hypothesis": HYPOTHESIS, "static_spec": STATIC_SPEC,
            "architectures": ARCHITECTURES, "nested_fold_audits": audits,
            "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V156", "status": "MATCH", "canonical_check_count": 14,
            "reported": selected["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
            "material_gate": gate,
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(evidence, out / "V156_HIERARCHICAL_MARKOV_RESPONSE_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V156_FAIL_CLOSED_SELECTED_OOF.csv.gz")
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
            "explicit mode/output conflicts with controller-bound no-argument V156 full run",
        )
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    elif args.output is None:
        args.output = DEFAULT_OUT
    if args.full_run:
        require(bool(CONTROLLER_OUTPUT), "V156 direct full run forbidden without MARKET_BIO_VERSION_OUTPUT")
        require(args.output.resolve() == Path(CONTROLLER_OUTPUT).resolve(), "V156 controller full-run output mismatch")
    return args


def main() -> None:
    args = parse_args()
    bounded = bool(args.audit_only or args.smoke_test or args.support_probe)
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
            "static_spec": STATIC_SPEC,
            "architecture": "equal-event-group first-order response transition posterior with fixed market to source-family to issuer hierarchical Beta shrinkage and unseen-issuer backoff",
            "initial_static_response_prior_rejected_due_collision": ["V40", "V61", "V71", "V73"],
            "causal_historical_response_only": True,
            "current_or_target_response_used": False,
            "micro_tuning_or_post_outer_change": False,
            "canonical_controller_gate_checks": 14,
            "controller_contract": {
                "no_argument_market_bio_version_output_full": True,
                "controller_output_exact_binding": True,
                "explicit_mode_or_output_conflict_fails_closed": True,
                "direct_full_without_controller_fails_closed": True,
                "research_staging_v156_compatible": True,
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
            require(args.smoke_market == "KR", "V156 support probe is reserved for KR:2")
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
