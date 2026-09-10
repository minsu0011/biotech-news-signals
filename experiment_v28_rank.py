"""Leakage-safe opened-history direction-rank search for the pure V34 seal.

Only opened V28-V33 outcomes are parsed. V34 SEAL rows are loaded as metadata
by ``load_search_frame`` and asserted to
have neither labels nor future returns before any candidate is evaluated.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import bio_news_30m_v3 as app
import runtime_limits


ROOT = Path(__file__).resolve().parent
CURRENT_LABEL = "v34"
RECENT = ("v32", "v33")
OLDER = ("v30", "v31")
SEALED_SOURCES = {"US_EXACT_V34_SEAL", "KR_EXACT_V34_SEAL"}
SETS = {
    "US": {
        "v28": ({"US_EXACT_V28_DEV"}, {"US_EXACT_V28_SEAL"}),
        "v29": ({"US_EXACT_V29_DEV"}, {"US_EXACT_V29_SEAL"}),
        "v30": ({"US_EXACT_V30_DEV"}, {"US_EXACT_V30_SEAL"}),
        "v31": ({"US_EXACT_V31_DEV"}, {"US_EXACT_V31_SEAL"}),
        "v32": ({"US_EXACT_V31_DEV", "US_EXACT_V31_SEAL"}, {"US_EXACT_V32_SEAL"}),
        "v33": ({"US_EXACT_V33_SEAL"}, {"US_EXACT_V33_SEAL"}),
    },
    "KR": {
        "v28": ({"KR_EXACT_V28_DEV"}, {"KR_EXACT_V28_SEAL"}),
        "v29": ({"KR_EXACT_V29_DEV"}, {"KR_EXACT_V29_SEAL"}),
        "v30": ({"KR_EXACT_V30_DEV"}, {"KR_EXACT_V30_SEAL"}),
        "v31": ({"KR_EXACT_V31_DEV"}, {"KR_EXACT_V31_SEAL"}),
        "v32": ({"KR_EXACT_V31_DEV", "KR_EXACT_V31_SEAL"}, {"KR_EXACT_V32_SEAL"}),
        "v33": ({"KR_EXACT_V33_SEAL"}, {"KR_EXACT_V33_SEAL"}),
    },
}
EXCLUDED = {
    "year", "month", "weekday", "hour", "minute", "minutes_from_open",
    "minutes_to_close", "event_hour", "event_weekday", "event_month",
    "event_hour_sin", "event_hour_cos", "headline_len", "body_len",
}
FEATURES = tuple(column for column in app.NUMERIC_COLS if column not in EXCLUDED)


def rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    fill = float(np.median(reference)) if len(reference) else 0.0
    values = np.nan_to_num(np.asarray(values, float), nan=fill, posinf=fill, neginf=fill)
    if not len(reference):
        return np.full(len(values), 0.5)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right + 1.0) / (2.0 * len(reference))


def metric(y: np.ndarray, probability: np.ndarray, cutoff: float,
           auc: float | None = None) -> dict:
    prediction = probability >= cutoff
    accuracy = float((prediction == y).mean())
    naive = max(float(y.mean()), 1.0 - float(y.mean()))
    positive = y == 1
    negative = ~positive
    balanced = 0.5 * (
        float(prediction[positive].mean())
        + float((~prediction[negative]).mean())
    )
    return {
        "n": int(len(y)),
        "balanced_accuracy": balanced,
        "auc": float(roc_auc_score(y, probability)) if auc is None else float(auc),
        "accuracy": accuracy,
        "edge": accuracy - naive,
        "pred_up": float(prediction.mean()),
    }


def best_row(market: str, components: list, scores: dict, labels: dict) -> dict:
    probability = {
        name: sum(
            float(weight) * (scores[name][column] if float(sign) > 0 else 1.0 - scores[name][column])
            for column, sign, weight in components
        ) / sum(float(item[2]) for item in components)
        for name in scores
    }
    aucs = {name: float(roc_auc_score(labels[name], value))
            for name, value in probability.items()}
    best = None
    for cutoff in np.arange(0.30, 0.701, 0.01):
        sets = {name: metric(labels[name], value, float(cutoff), aucs[name])
                for name, value in probability.items()}
        recent = tuple(sets[label] for label in RECENT)
        checks = sum(item["balanced_accuracy"] >= 0.54 for item in recent)
        checks += sum(item["auc"] >= 0.56 for item in recent)
        checks += sum(item["edge"] >= 0.02 for item in recent)
        checks += sum(sets[label]["balanced_accuracy"] >= 0.51 for label in OLDER)
        checks += sum(sets[label]["auc"] >= 0.52 for label in OLDER)
        score = (
            min(item["balanced_accuracy"] for item in recent)
            + min(item["auc"] for item in recent)
            + min(item["edge"] for item in recent)
            + 0.20 * min(sets[label]["balanced_accuracy"] for label in OLDER)
            + 0.20 * min(sets[label]["auc"] for label in OLDER)
        )
        row = {
            "market": market,
            "components": components,
            "cutoff": float(cutoff),
            "sets": sets,
            "checks": int(checks),
            "score": float(score),
        }
        if best is None or (row["checks"], row["score"]) > (best["checks"], best["score"]):
            best = row
    return best


def main() -> None:
    runtime_limits.configure()
    data = app.load_search_frame(app.Config())
    sealed = data.source.isin(SEALED_SOURCES)
    if (data.loc[sealed, "y"].notna().any()
            or data.loc[sealed, "fwd_ret_30m"].notna().any()):
        raise RuntimeError(f"{CURRENT_LABEL.upper()} seal outcomes exposed")
    opened = data[data.y.notna()].copy()
    output = {}
    for market in ("US", "KR"):
        scores = {}
        labels = {}
        sizes = {}
        for name, (reference_sources, target_sources) in SETS[market].items():
            reference = opened[
                opened.market.eq(market) & opened.source.isin(reference_sources)
            ].copy()
            target = opened[
                opened.market.eq(market) & opened.source.isin(target_sources)
            ].copy()
            if reference.empty or target.empty or target.y.nunique() < 2:
                raise RuntimeError(f"insufficient {market}/{name}: ref={len(reference)} target={len(target)}")
            reference_features = app.model_frame(reference)
            target_features = app.model_frame(target)
            labels[name] = target.y.astype(int).to_numpy()
            sizes[name] = {"reference": len(reference), "target": len(target)}
            scores[name] = {
                column: rank(
                    pd.to_numeric(reference_features[column], errors="coerce").to_numpy(float),
                    pd.to_numeric(target_features[column], errors="coerce").to_numpy(float),
                )
                for column in FEATURES
            }
        print("DATA", market, sizes, flush=True)
        singles = [
            best_row(market, [(column, sign, 1.0)], scores, labels)
            for column, sign in itertools.product(FEATURES, (-1.0, 1.0))
        ]
        singles.sort(key=lambda row: (row["checks"], row["score"]), reverse=True)
        chosen = []
        for row in singles:
            column, sign, _ = row["components"][0]
            if column not in {item[0] for item in chosen}:
                chosen.append((column, sign))
            if len(chosen) == 20:
                break
        rows = list(singles)
        for (first, first_sign), (second, second_sign) in itertools.combinations(chosen, 2):
            for weight in (0.20, 0.35, 0.50, 0.65, 0.80):
                rows.append(best_row(
                    market,
                    [(first, first_sign, weight), (second, second_sign, 1.0 - weight)],
                    scores, labels,
                ))
        for combination in itertools.combinations(chosen[:14], 3):
            rows.append(best_row(
                market,
                [(column, sign, 1.0 / 3.0) for column, sign in combination],
                scores, labels,
            ))
        rows.sort(key=lambda row: (row["checks"], row["score"]), reverse=True)
        output[market] = rows[:500]
        for row in rows[:20]:
            print("TOP", market, json.dumps(row, ensure_ascii=False), flush=True)
    path = ROOT / "cache" / f"{CURRENT_LABEL}_rank_search.json"
    path.write_text(json.dumps({
        f"{CURRENT_LABEL}_seal_outcomes_loaded": False,
        "runtime": runtime_limits.status(),
        "features": FEATURES,
        "rows": output,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT=" + str(path), flush=True)


if __name__ == "__main__":
    main()
