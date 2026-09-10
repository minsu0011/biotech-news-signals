"""GPU/32-thread supervised direction audit using no V34 SEAL outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app
import runtime_limits


ROOT = Path(__file__).resolve().parent
SOURCES = {
    "US": {
        "prior_dev": {"US_EXACT_V31_DEV", "US_EXACT_V31_SEAL"},
        "prior_audit": {"US_EXACT_V32_SEAL"}, "current_dev": {"US_EXACT_V33_SEAL"},
    },
    "KR": {
        "prior_dev": {"KR_EXACT_V31_DEV", "KR_EXACT_V31_SEAL"},
        "prior_audit": {"KR_EXACT_V32_SEAL"}, "current_dev": {"KR_EXACT_V33_SEAL"},
    },
}
SPECS = [
    {"kind": "native_cat", "iterations": 120, "depth": 3, "cat_cols": ["event_type"]},
    {"kind": "native_cat", "iterations": 220, "depth": 4, "cat_cols": ["event_type"]},
    {"kind": "lgb", "n_estimators": 120, "num_leaves": 7, "cat_cols": ["event_type"]},
    {"kind": "lgb", "n_estimators": 220, "num_leaves": 11, "cat_cols": ["event_type"]},
]


def groups(frame: pd.DataFrame, market: str) -> pd.Series:
    day = pd.to_datetime(frame.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    if market == "KR":
        # Exact duplicated disclosure/news records are already deduplicated;
        # ticker-day grouping is the conservative remaining dependence unit.
        return frame.ticker.astype(str) + "|" + day
    return frame.ticker.astype(str) + "|" + day


def metrics(y: np.ndarray, probability: np.ndarray, cutoff: float) -> dict:
    prediction = probability >= cutoff
    accuracy = float((prediction == y).mean())
    naive = max(float(y.mean()), 1.0 - float(y.mean()))
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auc": float(roc_auc_score(y, probability)),
        "accuracy": accuracy,
        "edge": accuracy - naive,
        "pred_up": float(prediction.mean()),
    }


def main() -> None:
    runtime_limits.configure()
    data = app.load_search_frame(app.Config())
    seal = data.source.isin({"US_EXACT_V34_SEAL", "KR_EXACT_V34_SEAL"})
    if data.loc[seal, ["y", "fwd_ret_30m"]].notna().any().any():
        raise RuntimeError("V34 seal outcomes exposed")
    opened = data[data.y.notna()].copy()
    rows = []
    for market in ("US", "KR"):
        source = SOURCES[market]
        current = opened[
            opened.market.eq(market) & opened.source.isin(source["current_dev"])
        ].sort_values("event_time_utc").reset_index(drop=True)
        prior = opened[
            opened.market.eq(market) & opened.source.isin(source["prior_dev"])
        ].sort_values("event_time_utc").reset_index(drop=True)
        audit = opened[
            opened.market.eq(market) & opened.source.isin(source["prior_audit"])
        ].sort_values("event_time_utc").reset_index(drop=True)
        if min(len(current), len(prior), len(audit)) == 0:
            raise RuntimeError(f"missing model audit partition for {market}")
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=app.SEED)
        folds = list(splitter.split(current, current.y.astype(int), groups(current, market)))
        for index, spec in enumerate(SPECS):
            oof = np.full(len(current), np.nan)
            for fold, (train, valid) in enumerate(folds, 1):
                model = app.make_model(spec)
                model.fit(app.model_frame(current.iloc[train]), current.y.iloc[train].astype(int))
                oof[valid] = model.predict_proba(app.model_frame(current.iloc[valid]))[:, 1]
                print(f"FIT {market} spec={index + 1}/{len(SPECS)} fold={fold}/4", flush=True)
            if not np.isfinite(oof).all():
                raise RuntimeError("incomplete V29 OOF")
            prior_model = app.make_model(spec)
            prior_model.fit(app.model_frame(prior), prior.y.astype(int))
            transferred = prior_model.predict_proba(app.model_frame(audit))[:, 1]
            best = None
            for invert in (False, True):
                current_probability = 1.0 - oof if invert else oof
                audit_probability = 1.0 - transferred if invert else transferred
                for cutoff in np.arange(0.30, 0.701, 0.01):
                    current_metrics = metrics(
                        current.y.astype(int).to_numpy(), current_probability, float(cutoff)
                    )
                    audit_metrics = metrics(
                        audit.y.astype(int).to_numpy(), audit_probability, float(cutoff)
                    )
                    checks = sum((
                        current_metrics["balanced_accuracy"] >= 0.54,
                        current_metrics["auc"] >= 0.56,
                        current_metrics["edge"] >= 0.02,
                        audit_metrics["balanced_accuracy"] >= 0.52,
                        audit_metrics["auc"] >= 0.54,
                        audit_metrics["edge"] >= 0.00,
                    ))
                    score = (
                        current_metrics["balanced_accuracy"] + current_metrics["auc"]
                        + current_metrics["edge"]
                        + 0.5 * audit_metrics["balanced_accuracy"]
                        + 0.5 * audit_metrics["auc"] + 0.25 * audit_metrics["edge"]
                    )
                    candidate = {
                        "market": market, "spec": spec, "invert": invert,
                        "cutoff": float(cutoff), "checks": int(checks),
                        "score": float(score),
                        "v33_oof": current_metrics, "v32_transfer": audit_metrics,
                    }
                    if best is None or (candidate["checks"], candidate["score"]) > (
                            best["checks"], best["score"]):
                        best = candidate
            rows.append(best)
            print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: (row["checks"], row["score"]), reverse=True)
    path = ROOT / "cache" / "v34_models.json"
    path.write_text(json.dumps({
        "v34_seal_outcomes_loaded": False,
        "runtime": runtime_limits.status(), "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT=" + str(path), flush=True)


if __name__ == "__main__":
    main()
