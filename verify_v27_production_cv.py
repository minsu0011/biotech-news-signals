"""Exact V27 experiment/production parity and opened-transfer audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import experiment_v27_confidence as experiment
import runtime_limits


ROOT = Path(__file__).resolve().parent
TARGET = {"US": 258.0, "KR": 325.0}
SOURCES = {
    "v26": {
        "US": ("SEC_V26_DEV", "SEC_V26_SEAL"),
        "KR": ("NAVER_NEWS_V26_DEV", "NAVER_NEWS_V26_SEAL"),
    },
    "v27": {
        "US": ("SEC_V27_DEV", "SEC_V27_DEV"),
        "KR": ("NAVER_NEWS_V27_DEV", "NAVER_NEWS_V27_DEV"),
    },
}


def assert_close(name: str, actual, expected, atol: float = 1e-12) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True):
        delta = float(np.nanmax(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{name} mismatch max_abs={delta}")


def main() -> None:
    runtime_limits.configure()
    cfg = app.Config()
    data = app.load_search_frame(cfg)
    seal_mask = data.source.isin({"SEC_V27_SEAL", "NAVER_NEWS_V27_SEAL"})
    if data.loc[seal_mask, "y"].notna().any() or data.loc[seal_mask, "fwd_ret_30m"].notna().any():
        raise RuntimeError("V27 seal outcome leakage")
    opened = data[data.y.notna()].copy()
    cache = json.loads((ROOT / "cache" / "v27_confidence.json").read_text(encoding="utf-8"))
    policy = next(
        row for row in cache["policies"]
        if row["source_index"] == 8
        and row["gates"]["US"]["column"] == "range_5m"
        and row["gates"]["KR"]["column"] == "entry_bar_range"
        and float(row["gates"]["KR"]["gate_quantile"]) == 0.825
    )
    results = {}
    fitted_v27 = {}
    for label, by_market in SOURCES.items():
        parts = []
        results[label] = {"markets": {}}
        for market, (reference_source, target_source) in by_market.items():
            reference_frame = opened[
                opened.market.eq(market) & opened.source.eq(reference_source)
            ].copy().sort_values("event_time_utc").reset_index(drop=True)
            target_frame = opened[
                opened.market.eq(market) & opened.source.eq(target_source)
            ].copy().sort_values("event_time_utc").reset_index(drop=True)
            reference = app.model_frame(reference_frame)
            target = app.model_frame(target_frame)
            spec = app.model_specs(market)[0]
            expected_spec = policy["markets"][market]
            assert spec["kind"] == "v27_stable_rank"
            assert spec["components"] == expected_spec["components"]
            assert_close(f"{market} cutoff", spec["direction_cutoff"], expected_spec["cutoff"])

            model = app.make_model(spec)
            model.fit(reference, reference_frame.y.astype(int))
            production_probability = model.predict_proba(target)[:, 1]
            production_eligible = model.eligible(target)

            reference_direction = experiment.direction(reference, reference, spec["components"])
            target_direction = experiment.direction(reference, target, spec["components"])
            experiment_probability = experiment.encode(
                reference, target, reference_direction, target_direction,
                float(spec["direction_cutoff"]), policy["gates"][market],
            )
            experiment_eligible = np.ones(len(target), dtype=bool)
            if market == "KR":
                experiment_eligible = experiment.eligibility(
                    reference, target, reference_direction, target_direction,
                    float(spec["direction_cutoff"]), policy["eligibility"],
                )
            if not np.allclose(production_probability, experiment_probability,
                               rtol=0.0, atol=1e-12, equal_nan=True):
                difference=np.abs(production_probability-experiment_probability)
                positions=np.argsort(difference)[-5:][::-1]
                print("PARITY_DIAGNOSTIC="+json.dumps({
                    "label":label,"market":market,
                    "spec":spec,"policy_gate":policy["gates"][market],
                    "max_direction_error":float(np.max(np.abs(
                        model.direction_score(target)-target_direction
                    ))),
                    "rows":[{
                        "position":int(position),
                        "event_id":str(target_frame.iloc[position].event_id),
                        "production":float(production_probability[position]),
                        "experiment":float(experiment_probability[position]),
                        "direction_production":float(model.direction_score(target)[position]),
                        "direction_experiment":float(target_direction[position]),
                    } for position in positions],
                },ensure_ascii=False),flush=True)
            assert_close(f"{label} {market} probability", production_probability,
                         experiment_probability)
            if not np.array_equal(production_eligible, experiment_eligible):
                raise AssertionError(f"{label} {market} eligibility mismatch")

            kept = target_frame.loc[production_eligible].reset_index(drop=True)
            probability = production_probability[production_eligible]
            part = kept[["event_id", "event_time_utc", "market", "ticker", "y", "fwd_ret_30m"]].copy()
            part["prob"] = probability
            if label == "v27":
                part["eval_weight"] = TARGET[market] / len(part)
                fitted_v27[market] = model
                oof_probability, oof_eligible = model.crossfit_output(reference)
                assert_close(f"{market} crossfit probability", oof_probability,
                             production_probability)
                if not np.array_equal(oof_eligible, production_eligible):
                    raise AssertionError(f"{market} crossfit eligibility mismatch")
            parts.append(part)
            results[label]["markets"][market] = {
                "reference_n": len(reference_frame),
                "target_n": len(target_frame),
                "eligible_n": int(production_eligible.sum()),
                "highconf_n": int((np.maximum(probability, 1.0 - probability) >= 0.925).sum()),
                "max_probability_abs_error": float(np.max(np.abs(
                    production_probability - experiment_probability
                ))),
            }
        combined = pd.concat(parts, ignore_index=True)
        metrics = app.evaluate(combined, 0.925, 0.002)
        expected_metrics = policy["sets"][label]["metrics"]
        for key in (
            "balanced_accuracy", "edge_vs_naive", "highconf_coverage",
            "highconf_accuracy", "strategy_mean_signed_net",
        ):
            assert_close(f"{label} metric {key}", metrics[key], expected_metrics[key])
        for market in ("US", "KR"):
            assert_close(f"{label} {market} balanced_accuracy",
                         metrics["by_market"][market]["balanced_accuracy"],
                         expected_metrics["by_market"][market]["balanced_accuracy"])
        results[label]["metrics"] = metrics
        results[label]["cached_preclip_auc"] = expected_metrics["auc"]
        results[label]["auc_clip_delta"] = metrics["auc"]-expected_metrics["auc"]
        results[label]["bootstrap"] = app.bootstrap_by_day(
            combined, 0.925, 0.002, nboot=2000
        )
        checks = [
            metrics["balanced_accuracy"] >= cfg.cert_min_bal_acc,
            metrics["auc"] >= cfg.cert_min_auc,
            metrics["edge_vs_naive"] >= cfg.cert_min_edge_vs_naive,
            metrics["highconf_coverage"] >= cfg.cert_min_highconf_coverage,
            metrics["highconf_accuracy"] >= cfg.cert_min_highconf_acc,
            metrics["strategy_mean_signed_net"] > cfg.cert_min_mean_signed_net,
            metrics["by_market"]["US"]["balanced_accuracy"] >= cfg.cert_min_market_bal_acc,
            metrics["by_market"]["KR"]["balanced_accuracy"] >= cfg.cert_min_market_bal_acc,
            results[label]["bootstrap"]["balanced_accuracy_lower95"] >= cfg.cert_min_boot_bal_lower95,
        ]
        results[label]["all_performance_checks_pass"] = bool(all(checks))
        if not all(checks):
            raise AssertionError(f"{label} clipped production policy failed checks: {checks}")

    # This is a label-free pre-seal count/probability check.  It neither loads
    # nor derives any V27 sealed outcome.
    sealed = data[seal_mask].copy().sort_values("event_time_utc").reset_index(drop=True)
    frozen = {
        "kind": "market_ensemble",
        "by_market": fitted_v27,
        "decision_cutoff_by_market": dict(cfg.decision_cutoff_by_market),
        "invert_probability_by_market": dict(cfg.invert_probability_by_market),
    }
    seal_probability = app.predict_proba_frozen(frozen, sealed)
    seal_eligible = app.predict_eligible_frozen(frozen, sealed)
    results["sealed_features_only"] = {
        "outcomes_loaded": False,
        "total_n": int(len(sealed)),
        "eligible_n": int(seal_eligible.sum()),
        "by_market_total": {
            market: int((sealed.market == market).sum()) for market in ("US", "KR")
        },
        "by_market_eligible": {
            market: int((seal_eligible & sealed.market.eq(market).to_numpy()).sum())
            for market in ("US", "KR")
        },
        "highconf_n": int((
            seal_eligible & (np.maximum(seal_probability, 1.0 - seal_probability) >= 0.925)
        ).sum()),
    }
    seal_counts=results["sealed_features_only"]
    if (seal_counts["eligible_n"] < cfg.cert_min_total_events or
            seal_counts["by_market_eligible"]["US"] < cfg.cert_min_us_events or
            seal_counts["by_market_eligible"]["KR"] < cfg.cert_min_kr_events):
        raise AssertionError(f"label-free sealed sample minimum failed: {seal_counts}")
    report = {
        "status": "PASS",
        "v27_seal_outcomes_loaded": False,
        "runtime": runtime_limits.status(),
        "selected_policy_source_index": policy["source_index"],
        "results": results,
    }
    path = ROOT / "cache" / "v27_production_verification.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("RESULT=" + str(path))


if __name__ == "__main__":
    main()
