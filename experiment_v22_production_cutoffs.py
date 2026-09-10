"""Production-ordered raw OOF cutoff scan for the selected V22 base models."""
from __future__ import annotations

import runtime_limits

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app
from verify_v22_production_cv import SPECS


ROOT = Path(__file__).resolve().parent
TARGET_WEIGHTS = {"US": 295.0, "KR": 578.0}


def groups(frame: pd.DataFrame, market: str) -> pd.Series:
    if market == "KR":
        return frame.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    day = pd.to_datetime(frame.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str) + "|" + day


def crossfit_raw(frame: pd.DataFrame, market: str, base_spec: dict) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=app.SEED)
    probability = np.full(len(frame), np.nan)
    for train_index, valid_index in splitter.split(frame, frame.y.astype(int), groups(frame, market)):
        model = app.make_model(base_spec)
        model.fit(app.model_frame(frame.iloc[train_index]), frame.y.iloc[train_index].astype(int))
        probability[valid_index] = model.predict_proba(app.model_frame(frame.iloc[valid_index]))[:, 1]
    return probability


def market_metric(frame: pd.DataFrame, probability: np.ndarray, cutoff: float) -> dict:
    y = frame.y.astype(int).to_numpy()
    pred = probability >= cutoff
    accuracy = float((pred == y).mean())
    up_rate = float(y.mean())
    return {
        "cutoff": float(cutoff), "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, probability)),
        "edge": accuracy - max(up_rate, 1 - up_rate),
        "up_prediction_rate": float(pred.mean()),
    }


def encoded(frame: pd.DataFrame, probability: np.ndarray, cutoff: float, market: str) -> pd.DataFrame:
    result = frame[[
        "event_id", "event_time_utc", "market", "ticker", "y", "fwd_ret_30m",
    ]].copy()
    result["prob"] = app.shift_probability(probability, cutoff)
    result["eval_weight"] = TARGET_WEIGHTS[market] / len(result)
    return result


def main() -> None:
    data = app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V22_SEAL", "NAVER_NEWS_V22_SEAL"}), "y"].notna().any():
        raise RuntimeError("V22 seal outcome exposure detected")
    frames = {
        "US": data[data.source.eq("SEC_V22_DEV") & data.y.notna()].copy()
                  .sort_values("event_time_utc").reset_index(drop=True),
        "KR": data[data.source.eq("NAVER_NEWS_V22_DEV") & data.y.notna()].copy()
                  .sort_values("event_time_utc").reset_index(drop=True),
    }
    raw = {
        market: crossfit_raw(frame, market, SPECS[market]["base_spec"])
        for market, frame in frames.items()
    }
    cache_parts = []
    for market, frame in frames.items():
        part = frame[["event_id", "market", "source", "event_time_utc", "ticker", "y", "fwd_ret_30m"]].copy()
        part["raw_probability"] = raw[market]
        cache_parts.append(part)
    pd.concat(cache_parts, ignore_index=True).to_parquet(
        ROOT / "cache" / "v22_production_raw_oof.parquet", index=False,
    )

    options = {
        "US": [market_metric(frames["US"], raw["US"], cutoff)
               for cutoff in np.arange(.50, .751, .005)],
        "KR": [market_metric(frames["KR"], raw["KR"], cutoff)
               for cutoff in np.arange(.45, .751, .005)],
    }
    for market in options:
        options[market].sort(
            key=lambda row: (
                int(row["balanced_accuracy"] >= .54) + int(row["edge"] >= .02),
                5 * row["balanced_accuracy"] + 2 * row["edge"],
            ), reverse=True,
        )

    pairs = []
    for us in options["US"][:30]:
        for kr in options["KR"][:30]:
            combined = pd.concat([
                encoded(frames["US"], raw["US"], us["cutoff"], "US"),
                encoded(frames["KR"], raw["KR"], kr["cutoff"], "KR"),
            ], ignore_index=True)
            metric = app.evaluate(combined, .60, .002)
            pass_count = sum((
                metric["balanced_accuracy"] >= .54,
                metric["edge_vs_naive"] >= .02,
                metric["by_market"]["US"]["balanced_accuracy"] >= .51,
                metric["by_market"]["KR"]["balanced_accuracy"] >= .51,
            ))
            score = 8 * metric["balanced_accuracy"] + 5 * metric["edge_vs_naive"]
            pairs.append({
                "pass_count": int(pass_count), "score": float(score),
                "us": us, "kr": kr,
                "combined": {
                    key: metric[key] for key in (
                        "accuracy", "balanced_accuracy", "auc", "up_rate",
                        "naive_accuracy", "edge_vs_naive", "by_market",
                    )
                },
            })
    pairs.sort(key=lambda row: (row["pass_count"], row["score"]), reverse=True)
    result = {
        "seal_outcomes_loaded": False, "runtime": runtime_limits.status(),
        "market_options": options, "pairs": pairs[:100],
    }
    path = ROOT / "cache" / "v22_production_cutoff_audits.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUNTIME=" + json.dumps(runtime_limits.status()))
    for market in ("US", "KR"):
        print(f"[{market}]")
        for row in options[market][:12]:
            print(json.dumps(row))
    print("[PAIRS]")
    for row in pairs[:15]:
        print(json.dumps(row))
    print("RESULT=" + str(path))


if __name__ == "__main__":
    main()
