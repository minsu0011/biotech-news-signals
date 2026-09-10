"""Exact transfer-policy parity and feature-only audit before freezing V35."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
SELECTED_FINALIST_INDEX = 13
V30 = {
    "US": ({"US_EXACT_V30_DEV"}, {"US_EXACT_V30_SEAL"}),
    "KR": ({"KR_EXACT_V30_DEV"}, {"KR_EXACT_V30_SEAL"}),
}


def prediction_frame(frame: pd.DataFrame, probability) -> pd.DataFrame:
    output = frame[[
        "event_id", "event_time_utc", "market", "ticker", "y", "fwd_ret_30m"
    ]].copy()
    output["prob"] = probability
    return output


def checks(metrics: dict, bootstrap: dict, cfg: app.Config) -> dict:
    return {
        "balanced_accuracy": metrics["balanced_accuracy"] >= cfg.cert_min_bal_acc,
        "auc": metrics["auc"] >= cfg.cert_min_auc,
        "edge_vs_naive": metrics["edge_vs_naive"] >= cfg.cert_min_edge_vs_naive,
        "highconf_coverage": metrics["highconf_coverage"] >= cfg.cert_min_highconf_coverage,
        "highconf_accuracy": metrics["highconf_accuracy"] >= cfg.cert_min_highconf_acc,
        "strategy_mean_signed_net": metrics["strategy_mean_signed_net"] > 0,
        "bootstrap_lower95": (
            bootstrap["balanced_accuracy_lower95"] >= cfg.cert_min_boot_bal_lower95
        ),
        "US_balanced_accuracy": (
            metrics["by_market"]["US"]["balanced_accuracy"]
            >= cfg.cert_min_market_bal_acc
        ),
        "KR_balanced_accuracy": (
            metrics["by_market"]["KR"]["balanced_accuracy"]
            >= cfg.cert_min_market_bal_acc
        ),
    }


def main() -> None:
    cfg = app.Config()
    data = app.load_search_frame(cfg)
    sealed = data.source.isin({"US_EXACT_V35_SEAL", "KR_EXACT_V35_SEAL"})
    if data.loc[sealed, ["y", "fwd_ret_30m"]].notna().any().any():
        raise RuntimeError("V35 seal outcomes exposed")
    opened = data[data.y.notna()].copy()

    forbidden = {"y", "fwd_ret_30m", "exit_price", "exit_time_utc"}
    sealed_features = pd.read_csv(
        app.LABELED_FILE, compression="gzip", dtype={"ticker": str}, low_memory=False,
        usecols=lambda column: column not in forbidden,
    )
    if forbidden & set(sealed_features.columns):
        raise RuntimeError("sealed feature audit parsed a forbidden outcome column")
    sealed_features = sealed_features[
        sealed_features.source.isin({"US_EXACT_V35_SEAL", "KR_EXACT_V35_SEAL"})
    ].copy()
    if len(sealed_features) != int(sealed.sum()):
        raise RuntimeError("sealed feature-only row count mismatch")

    current_parts = [
        app.cv_market_predictions(opened, market, app.model_specs(market)[0], cfg)
        for market in ("US", "KR")
    ]
    current = pd.concat(current_parts, ignore_index=True)
    current_metrics = app.evaluate(current, 0.925, cfg.round_trip_cost)
    current_bootstrap = app.bootstrap_by_day(
        current, 0.925, cfg.round_trip_cost, nboot=2000
    )

    feature_only_probabilities = []
    for market in ("US", "KR"):
        final_sources = {
            source for source_market, source in cfg.final_training_sources
            if source_market == market
        }
        reference = opened[
            opened.market.eq(market)
            & opened.source.isin(final_sources)
        ].sort_values("event_time_utc").reset_index(drop=True)
        target = sealed_features[
            sealed_features.market.eq(market)
        ].sort_values("event_time_utc").reset_index(drop=True)
        model = app.make_model(app.model_specs(market)[0])
        model.fit(app.model_frame(reference), reference.y.astype(int))
        probability = model.predict_proba(app.model_frame(target))[:, 1]
        feature_only_probabilities.extend(probability.tolist())
    feature_only_highconf = [
        max(probability, 1.0 - probability) >= 0.925
        for probability in feature_only_probabilities
    ]
    feature_only_audit = {
        "forbidden_outcome_columns_loaded": False,
        "n": len(feature_only_probabilities),
        "probabilities_finite_and_bounded": all(
            0.0 <= probability <= 1.0 for probability in feature_only_probabilities
        ),
        "predicted_highconf_n": int(sum(feature_only_highconf)),
        "predicted_highconf_coverage": float(sum(feature_only_highconf) / len(feature_only_highconf)),
    }
    if (not feature_only_audit["probabilities_finite_and_bounded"]
            or feature_only_audit["predicted_highconf_coverage"] < cfg.cert_min_highconf_coverage):
        raise RuntimeError(f"sealed feature-only audit failed: {feature_only_audit}")

    # Exact agreement with the selected pre-freeze search result is required.
    selected = json.loads(
        (ROOT / "cache" / "v35_confidence.json").read_text(encoding="utf-8")
    )["policies"][SELECTED_FINALIST_INDEX]["sets"]["v34"]["metrics"]
    parity_fields = (
        "balanced_accuracy", "auc", "edge_vs_naive", "highconf_coverage",
        "highconf_accuracy", "strategy_mean_signed_net",
    )
    parity_delta = {
        field: abs(float(current_metrics[field]) - float(selected[field]))
        for field in parity_fields
    }
    if max(parity_delta.values()) > 1e-12:
        raise RuntimeError(f"search/production probability parity failed: {parity_delta}")

    # Opened V30 is retained as a distribution-shift audit. It is reported,
    # not used as a substitute for the predeclared V31 transfer acceptance test.
    audit_parts = []
    for market, (reference_sources, target_sources) in V30.items():
        reference = opened[
            opened.market.eq(market) & opened.source.isin(reference_sources)
        ].sort_values("event_time_utc").reset_index(drop=True)
        target = opened[
            opened.market.eq(market) & opened.source.isin(target_sources)
        ].sort_values("event_time_utc").reset_index(drop=True)
        model = app.make_model(app.model_specs(market)[0])
        model.fit(app.model_frame(reference), reference.y.astype(int))
        probability = model.predict_proba(app.model_frame(target))[:, 1]
        part = prediction_frame(target, probability)
        weight_target = float(dict(cfg.dev_market_weight_targets)[market])
        part["eval_weight"] = weight_target / len(part)
        audit_parts.append(part)
    audit = pd.concat(audit_parts, ignore_index=True)
    audit_metrics = app.evaluate(audit, 0.925, cfg.round_trip_cost)
    audit_bootstrap = app.bootstrap_by_day(
        audit, 0.925, cfg.round_trip_cost, nboot=1000
    )

    current_checks = checks(current_metrics, current_bootstrap, cfg)
    if not all(current_checks.values()):
        raise RuntimeError(f"production parity failed: v34_transfer {current_checks}")

    report = {
        "status": "PASS",
        "v35_seal_outcomes_loaded": False,
        "model_specs": {market: app.model_specs(market)[0] for market in ("US", "KR")},
        "search_production_max_abs_delta": float(max(parity_delta.values())),
        "search_production_delta": parity_delta,
        "selected_finalist_index": SELECTED_FINALIST_INDEX,
        "v35_seal_feature_only_audit": feature_only_audit,
        "checks": {"v34_transfer": current_checks},
        "v30_opened_distribution_audit": {
            "metrics": audit_metrics, "bootstrap": audit_bootstrap,
        },
        "v34_transfer": {"metrics": current_metrics, "bootstrap": current_bootstrap},
    }
    path = ROOT / "output" / "V35" / "PRODUCTION_PARITY.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
