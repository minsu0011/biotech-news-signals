"""Outcome-free V32 feature coverage audit for precomputed V31 finalists."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
FORBIDDEN = {"y", "fwd_ret_30m", "exit_price", "exit_time_utc"}
SEALED = {"US_EXACT_V32_SEAL", "KR_EXACT_V32_SEAL"}


def production_spec(market: str, policy: dict) -> dict:
    direction = policy["direction"]
    gate = policy["gate"]
    return {
        "kind": "v32_stable_rank",
        "market": market,
        "reference_sources": [f"{market}_EXACT_V31_DEV"],
        "direction_cutoff": float(direction["cutoff"]),
        "components": direction["components"],
        "opportunity_column": gate["column"],
        "opportunity_sign": float(gate["sign"]),
        "margin_weight": float(gate["margin_weight"]),
        "gate_quantile": float(gate["gate_quantile"]),
    }


def main() -> None:
    cfg = app.Config()
    opened = app.load_search_frame(cfg)
    opened = opened[opened.y.notna()].copy()
    features = pd.read_csv(
        app.LABELED_FILE,
        compression="gzip",
        dtype={"ticker": str},
        low_memory=False,
        usecols=lambda column: column not in FORBIDDEN,
    )
    if FORBIDDEN & set(features.columns):
        raise RuntimeError("feature-only audit parsed a forbidden outcome column")
    features = features[features.source.isin(SEALED)].copy()
    candidates = json.loads(
        (ROOT / "cache" / "v32_confidence.json").read_text(encoding="utf-8")
    )["policies"][:30]

    rows = []
    for index, candidate in enumerate(candidates):
        probabilities = []
        by_market = {}
        for market in ("US", "KR"):
            final_sources = {
                source for source_market, source in cfg.final_training_sources
                if source_market == market
            }
            reference = opened[
                opened.market.eq(market) & opened.source.isin(final_sources)
            ].sort_values("event_time_utc").reset_index(drop=True)
            target = features[features.market.eq(market)].sort_values(
                "event_time_utc"
            ).reset_index(drop=True)
            spec = production_spec(market, candidate["policies"][market])
            model = app.make_model(spec)
            model.fit(app.model_frame(reference), reference.y.astype(int))
            probability = model.predict_proba(app.model_frame(target))[:, 1]
            if not np.isfinite(probability).all():
                raise RuntimeError(f"nonfinite finalist probability: {index}/{market}")
            high = np.maximum(probability, 1.0 - probability) >= 0.925
            probabilities.append(probability)
            by_market[market] = {
                "n": int(len(probability)),
                "highconf_n": int(high.sum()),
                "highconf_coverage": float(high.mean()),
            }
        combined = np.concatenate(probabilities)
        high = np.maximum(combined, 1.0 - combined) >= 0.925
        rows.append({
            "finalist_index": index,
            "precomputed_checks": int(candidate["checks"]),
            "precomputed_score": float(candidate["score"]),
            "feature_only_n": int(len(combined)),
            "feature_only_highconf_n": int(high.sum()),
            "feature_only_highconf_coverage": float(high.mean()),
            "probabilities_finite_and_bounded": bool(
                np.isfinite(combined).all()
                and (combined >= 0.0).all()
                and (combined <= 1.0).all()
            ),
            "by_market": by_market,
        })

    eligible = [
        row for row in rows
        if row["precomputed_checks"] == 9
        and row["probabilities_finite_and_bounded"]
        and row["feature_only_highconf_coverage"] >= cfg.cert_min_highconf_coverage
    ]
    if not eligible:
        raise RuntimeError("no precomputed finalist clears feature-only coverage")
    selected = eligible[0]
    report = {
        "status": "PASS",
        "v32_seal_outcomes_loaded": False,
        "forbidden_outcome_columns_loaded": False,
        "selection_rule": (
            "first precomputed V31 finalist with all nine opened checks and "
            "V32 feature-only high-confidence coverage >= certification minimum"
        ),
        "selected_finalist_index": selected["finalist_index"],
        "selected": selected,
        "rows": rows,
    }
    path = ROOT / "output" / "V32" / "FEATURE_ONLY_FINALIST_AUDIT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
