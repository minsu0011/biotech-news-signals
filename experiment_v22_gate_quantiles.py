"""Tune only the V22 high-confidence gate quantiles on production OOF."""
from __future__ import annotations

import runtime_limits

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app
from verify_v22_production_cv import SPECS


ROOT = Path(__file__).resolve().parent
QUANTILES = (.75, .775, .80, .825, .85)
TARGET_WEIGHTS = {"US": 295.0, "KR": 578.0}


def group_values(frame: pd.DataFrame, market: str) -> pd.Series:
    if market == "KR":
        return frame.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    day = pd.to_datetime(frame.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str) + "|" + day


def prediction_frame(frame: pd.DataFrame, probability: np.ndarray,
                     target_weight: float | None = None) -> pd.DataFrame:
    result = frame[[
        "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
    ]].copy()
    result["prob"] = probability
    if target_weight is not None:
        result["eval_weight"] = target_weight / len(result)
    return result


def crossfit_quantiles(frame: pd.DataFrame, market: str, spec: dict) -> dict[float, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=app.SEED)
    probability = {quantile: np.full(len(frame), np.nan) for quantile in QUANTILES}
    for train_index, valid_index in splitter.split(
        frame, frame.y.astype(int), group_values(frame, market),
    ):
        model = app.make_model(copy.deepcopy(spec))
        model.fit(app.model_frame(frame.iloc[train_index]), frame.y.iloc[train_index].astype(int))
        valid_features = app.model_frame(frame.iloc[valid_index])
        for quantile in QUANTILES:
            model.gate_quantile = quantile
            probability[quantile][valid_index] = model.predict_proba(valid_features)[:, 1]
    return {
        quantile: prediction_frame(frame, values, TARGET_WEIGHTS[market])
        for quantile, values in probability.items()
    }


def main() -> None:
    cfg = app.Config()
    data = app.load_search_frame(cfg)
    if data.loc[data.source.isin({"SEC_V22_SEAL", "NAVER_NEWS_V22_SEAL"}), "y"].notna().any():
        raise RuntimeError("V22 seal outcome exposure detected")
    target = {
        "US": data[data.source.eq("SEC_V22_DEV") & data.y.notna()].copy()
                  .sort_values("event_time_utc").reset_index(drop=True),
        "KR": data[data.source.eq("NAVER_NEWS_V22_DEV") & data.y.notna()].copy()
                  .sort_values("event_time_utc").reset_index(drop=True),
    }
    oof = {
        market: crossfit_quantiles(target[market], market, SPECS[market])
        for market in ("US", "KR")
    }

    rows = []
    for us_quantile, kr_quantile in itertools.product(QUANTILES, repeat=2):
        combined = pd.concat([
            oof["US"][us_quantile], oof["KR"][kr_quantile],
        ], ignore_index=True)
        metrics = app.evaluate(combined, .925, .002)
        bootstrap = app.bootstrap_by_day(combined, .925, .002, nboot=600)
        required = [
            metrics["balanced_accuracy"] >= .54, metrics["auc"] >= .56,
            metrics["edge_vs_naive"] >= .02, metrics["highconf_coverage"] >= .15,
            metrics["highconf_accuracy"] >= .60, metrics["strategy_mean_signed_net"] > 0,
            bootstrap["balanced_accuracy_lower95"] >= .50,
            metrics["by_market"]["US"]["balanced_accuracy"] >= .51,
            metrics["by_market"]["KR"]["balanced_accuracy"] >= .51,
        ]
        score = (
            4 * metrics["balanced_accuracy"] + 2 * metrics["auc"]
            + 3 * metrics["highconf_accuracy"] + 40 * metrics["strategy_mean_signed_net"]
            + .2 * metrics["highconf_coverage"]
        )
        rows.append({
            "us_quantile": us_quantile, "kr_quantile": kr_quantile,
            "pass_count": int(sum(required)), "score": float(score),
            "metrics": metrics, "bootstrap": bootstrap,
        })
    rows.sort(key=lambda row: (row["pass_count"], row["score"]), reverse=True)
    result = {
        "seal_outcomes_loaded": False, "runtime": runtime_limits.status(),
        "rows": rows,
    }
    path = ROOT / "cache" / "v22_gate_quantile_audits.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUNTIME=" + json.dumps(runtime_limits.status()))
    for row in rows[:15]:
        metric = row["metrics"]
        print(
            f"q_us={row['us_quantile']:.3f} q_kr={row['kr_quantile']:.3f} "
            f"passes={row['pass_count']}/9 bal={metric['balanced_accuracy']:.4f} "
            f"auc={metric['auc']:.4f} edge={metric['edge_vs_naive']:.4f} "
            f"hc={metric['highconf_accuracy']:.4f}/{metric['highconf_coverage']:.4f} "
            f"net={metric['strategy_mean_signed_net']:.5f} "
            f"boot={row['bootstrap']['balanced_accuracy_lower95']:.4f}",
        )
    print("RESULT=" + str(path))


if __name__ == "__main__":
    main()
