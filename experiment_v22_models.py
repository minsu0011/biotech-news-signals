"""V22 grouped-OOF direction scan; V22 seal outcomes are never loaded."""
from __future__ import annotations

import runtime_limits

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent


def candidate_specs() -> list[dict]:
    return [
        {"kind": "native_cat", "iterations": 220, "depth": depth,
         "cat_cols": ["event_type", "ticker"]}
        for depth in (2, 3, 4, 5, 6)
    ] + [
        {"kind": "lgb", "n_estimators": 220, "num_leaves": leaves,
         "cat_cols": ["event_type", "ticker"]}
        for leaves in (5, 9, 15)
    ] + [
        {"kind": "logit", "C": c, "class_weight": "balanced"}
        for c in (.003, .01, .03, .1)
    ]


def groups(frame: pd.DataFrame, market: str) -> pd.Series:
    if market == "KR":
        return frame.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    day = pd.to_datetime(frame.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    return frame.ticker.astype(str) + "|" + day


def crossfit(frame: pd.DataFrame, market: str, spec: dict) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=app.SEED,
    )
    probability = np.full(len(frame), np.nan)
    for train_index, valid_index in splitter.split(frame, frame.y.astype(int), groups(frame, market)):
        model = app.make_model(spec)
        model.fit(app.model_frame(frame.iloc[train_index]), frame.y.iloc[train_index].astype(int))
        probability[valid_index] = model.predict_proba(app.model_frame(frame.iloc[valid_index]))[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError("missing OOF probabilities")
    return probability


def metrics(frame: pd.DataFrame, probability: np.ndarray, cutoff: float) -> dict:
    y = frame.y.astype(int).to_numpy()
    pred = (probability >= cutoff).astype(int)
    accuracy = float((pred == y).mean())
    up_rate = float(y.mean())
    return {
        "n": int(len(frame)),
        "up_rate": up_rate,
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, probability)),
        "edge": accuracy - max(up_rate, 1.0 - up_rate),
        "up_prediction_rate": float(pred.mean()),
    }


def choose(audits: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> list[dict]:
    rows = []
    for cutoff in np.arange(.30, .751, .005):
        result = {
            name: metrics(frame, probability, float(cutoff))
            for name, (frame, probability) in audits.items()
        }
        core = list(result.values())
        pass_count = sum(value["balanced_accuracy"] >= .51 for value in core)
        pass_count += sum(value["edge"] >= 0 for value in core)
        score = (
            6.0 * result["v22_oof"]["balanced_accuracy"]
            + 5.0 * min(value["balanced_accuracy"] for value in core)
            + 3.0 * result["v22_oof"]["edge"]
            + min(value["edge"] for value in core)
        )
        rows.append({
            "pass_count": int(pass_count), "score": float(score),
            "cutoff": float(cutoff), "audits": result,
        })
    return sorted(rows, key=lambda row: (row["pass_count"], row["score"]), reverse=True)


def main() -> None:
    cfg = app.Config()
    data = app.load_search_frame(cfg)
    sealed = data.source.isin({"SEC_V22_SEAL", "NAVER_NEWS_V22_SEAL"})
    if data.loc[sealed, "y"].notna().any():
        raise RuntimeError("V22 seal outcome exposure detected")
    print("RUNTIME=" + json.dumps(runtime_limits.status()))

    targets = {
        "US": data[data.source.eq("SEC_V22_DEV") & data.y.notna()].copy().reset_index(drop=True),
        "KR": data[data.source.eq("NAVER_NEWS_V22_DEV") & data.y.notna()].copy().reset_index(drop=True),
    }
    external_sources = {
        "US": ("SEC_V19_SEAL", "SEC_V21_SEAL"),
        "KR": ("NAVER_NEWS_V21_SEAL",),
    }
    all_predictions = []
    output = {
        "seal_outcomes_loaded": False,
        "runtime": runtime_limits.status(),
        "markets": {},
    }
    for market, target in targets.items():
        market_rows = []
        for index, spec in enumerate(candidate_specs()):
            name = f"m{index:02d}"
            oof = crossfit(target, market, spec)
            model = app.make_model(spec)
            model.fit(app.model_frame(target), target.y.astype(int))
            audits: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
                "v22_oof": (target, oof),
            }
            for source in external_sources[market]:
                audit = data[data.source.eq(source) & data.y.notna()].copy().reset_index(drop=True)
                audits[source] = (
                    audit, model.predict_proba(app.model_frame(audit))[:, 1],
                )
            choices = choose(audits)
            row = {
                "name": name, "spec": spec,
                "auc_by_audit": {
                    audit_name: audit_metrics["auc"]
                    for audit_name, audit_metrics in choices[0]["audits"].items()
                },
                "choices": choices[:12],
            }
            market_rows.append(row)
            print(
                f"[{market} {name}] cutoff={choices[0]['cutoff']:.3f} "
                + " ".join(
                    f"{audit_name}:bal={value['balanced_accuracy']:.4f},"
                    f"auc={value['auc']:.4f},edge={value['edge']:.4f}"
                    for audit_name, value in choices[0]["audits"].items()
                ),
                flush=True,
            )
            prediction_frame = target[[
                "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
            ]].copy()
            prediction_frame["model_name"] = name
            prediction_frame["probability"] = oof
            all_predictions.append(prediction_frame)
        market_rows.sort(
            key=lambda row: (
                row["choices"][0]["pass_count"], row["choices"][0]["score"],
            ), reverse=True,
        )
        output["markets"][market] = {
            "n": int(len(target)), "up_rate": float(target.y.mean()),
            "models": market_rows,
        }
    pd.concat(all_predictions, ignore_index=True).to_parquet(
        ROOT / "cache" / "v22_oof_predictions.parquet", index=False,
    )
    (ROOT / "cache" / "v22_model_audits.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("RESULT=" + str(ROOT / "cache" / "v22_model_audits.json"))


if __name__ == "__main__":
    main()
