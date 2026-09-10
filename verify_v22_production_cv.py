"""Reproduce the selected V22 policy through production CV and prior audits."""
from __future__ import annotations

import runtime_limits

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent

SPECS = {
    "US": {
        "kind": "v22_supervised_confidence", "market": "US", "raw_cutoff": .585,
        "base_spec": {
            "kind": "native_cat", "iterations": 220, "depth": 6,
            "cat_cols": ["event_type", "ticker"],
        },
        "opportunity_column": "abs_pre_ret_5", "opportunity_sign": 1.0,
        "margin_weight": .25, "gate_quantile": .80,
    },
    "KR": {
        "kind": "v22_supervised_confidence", "market": "KR", "raw_cutoff": .575,
        "base_spec": {
            "kind": "blend",
            "components": [
                {"kind": "native_cat", "iterations": 220, "depth": depth,
                 "cat_cols": ["event_type", "ticker"]}
                for depth in (3, 4, 5)
            ],
            "weights": [1, 1, 1],
        },
        "opportunity_column": "volume_ratio_2_30", "opportunity_sign": -1.0,
        "margin_weight": .50, "gate_quantile": .75,
    },
}


def split_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    event_hash = frame.event_id.astype(str).map(
        lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % 2,
    ).to_numpy()
    ticker_hash = frame.ticker.astype(str).map(
        lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % 2,
    ).to_numpy()
    time_rank = pd.to_datetime(frame.event_time_utc, utc=True).rank(method="first", pct=True).to_numpy()
    return {
        "all": np.ones(len(frame), dtype=bool),
        "event_hash_0": event_hash == 0, "event_hash_1": event_hash == 1,
        "time_0": time_rank <= .5, "time_1": time_rank > .5,
        "ticker_hash_0": ticker_hash == 0, "ticker_hash_1": ticker_hash == 1,
    }


def audits(frame: pd.DataFrame) -> dict:
    return {
        name: app.evaluate(frame.loc[use].copy(), .925, .002)
        for name, use in split_masks(frame).items()
        if use.sum() >= 30 and frame.loc[use, "y"].nunique() > 1
    }


def prediction_frame(frame: pd.DataFrame, probability: np.ndarray,
                     weight_target: float | None = None) -> pd.DataFrame:
    result = frame[[
        "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
    ]].copy()
    result["prob"] = probability
    if weight_target is not None:
        result["eval_weight"] = weight_target / len(result)
    return result


def main() -> None:
    cfg = app.Config(dev_market_weight_targets=(("KR", 578.0), ("US", 295.0)))
    data = app.load_search_frame(cfg)
    if data.loc[data.source.isin({"SEC_V22_SEAL", "NAVER_NEWS_V22_SEAL"}), "y"].notna().any():
        raise RuntimeError("V22 seal outcome exposure detected")
    target = {
        "US": data[data.source.eq("SEC_V22_DEV") & data.y.notna()].copy().reset_index(drop=True),
        "KR": data[data.source.eq("NAVER_NEWS_V22_DEV") & data.y.notna()].copy().reset_index(drop=True),
    }

    production_oof = {
        market: app.cv_market_predictions(data, market, spec, cfg)
        for market, spec in SPECS.items()
    }
    combined = pd.concat([production_oof["US"], production_oof["KR"]], ignore_index=True)
    output = {
        "seal_outcomes_loaded": False,
        "runtime": runtime_limits.status(),
        "specs": SPECS,
        "production_cv": {
            "metrics": app.evaluate(combined, .925, .002),
            "bootstrap": app.bootstrap_by_day(combined, .925, .002, nboot=600),
            "subgroups": audits(combined),
        },
        "external": {},
    }

    fitted = {}
    for market, spec in SPECS.items():
        fitted[market] = app.make_model(spec)
        fitted[market].fit(app.model_frame(target[market]), target[market].y.astype(int))

    source_audits = {
        "US": ("SEC_V19_SEAL", "SEC_V21_SEAL"),
        "KR": ("NAVER_NEWS_V21_SEAL",),
    }
    external_frames = {}
    for market, sources in source_audits.items():
        for source in sources:
            frame = data[data.source.eq(source) & data.y.notna()].copy().reset_index(drop=True)
            probability = fitted[market].predict_proba(app.model_frame(frame))[:, 1]
            prediction = prediction_frame(frame, probability)
            external_frames[source] = prediction
            output["external"][source] = {
                "metrics": app.evaluate(prediction, .925, .002),
                "bootstrap": app.bootstrap_by_day(prediction, .925, .002, nboot=300),
            }

    v21 = pd.concat([
        external_frames["SEC_V21_SEAL"], external_frames["NAVER_NEWS_V21_SEAL"],
    ], ignore_index=True)
    output["external"]["V21_COMBINED"] = {
        "metrics": app.evaluate(v21, .925, .002),
        "bootstrap": app.bootstrap_by_day(v21, .925, .002, nboot=300),
    }

    path = ROOT / "cache" / "v22_production_verification.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUNTIME=" + json.dumps(runtime_limits.status()))
    print("PRODUCTION_CV=" + json.dumps(output["production_cv"], ensure_ascii=False))
    print("EXTERNAL=" + json.dumps(output["external"], ensure_ascii=False))
    print("RESULT=" + str(path))


if __name__ == "__main__":
    main()
