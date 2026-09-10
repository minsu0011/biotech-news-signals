"""Opened-history confidence-gate search with untouched pure V34 SEAL outcomes."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import runtime_limits


ROOT = Path(__file__).resolve().parent
CURRENT_LABEL = "v34"
PRIMARY_LABEL = "v33"
RECENT = ("v32", "v33")
OLDER = ("v30", "v31")
SEALED_SOURCES = {"US_EXACT_V34_SEAL", "KR_EXACT_V34_SEAL"}
SETS = {
    "v28": {
        "US": ({"US_EXACT_V28_DEV"}, {"US_EXACT_V28_SEAL"}),
        "KR": ({"KR_EXACT_V28_DEV"}, {"KR_EXACT_V28_SEAL"}),
    },
    "v29": {
        "US": ({"US_EXACT_V29_DEV"}, {"US_EXACT_V29_SEAL"}),
        "KR": ({"KR_EXACT_V29_DEV"}, {"KR_EXACT_V29_SEAL"}),
    },
    "v30": {
        "US": ({"US_EXACT_V30_DEV"}, {"US_EXACT_V30_SEAL"}),
        "KR": ({"KR_EXACT_V30_DEV"}, {"KR_EXACT_V30_SEAL"}),
    },
    "v31": {
        "US": ({"US_EXACT_V31_DEV"}, {"US_EXACT_V31_SEAL"}),
        "KR": ({"KR_EXACT_V31_DEV"}, {"KR_EXACT_V31_SEAL"}),
    },
    "v32": {
        "US": ({"US_EXACT_V31_DEV", "US_EXACT_V31_SEAL"}, {"US_EXACT_V32_SEAL"}),
        "KR": ({"KR_EXACT_V31_DEV", "KR_EXACT_V31_SEAL"}, {"KR_EXACT_V32_SEAL"}),
    },
    "v33": {
        "US": ({"US_EXACT_V33_SEAL"}, {"US_EXACT_V33_SEAL"}),
        "KR": ({"KR_EXACT_V33_SEAL"}, {"KR_EXACT_V33_SEAL"}),
    },
}
EXCLUDED = {
    "year", "month", "weekday", "hour", "minute", "minutes_from_open",
    "minutes_to_close", "event_hour", "event_weekday", "event_month",
    "event_hour_sin", "event_hour_cos", "headline_len", "body_len",
}
GATES = tuple(column for column in app.NUMERIC_COLS if column not in EXCLUDED)


def rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    fill = float(np.median(reference)) if len(reference) else 0.0
    values = np.nan_to_num(np.asarray(values, float), nan=fill, posinf=fill, neginf=fill)
    if not len(reference):
        return np.full(len(values), 0.5)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right + 1.0) / (2.0 * len(reference))


def values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def direction(reference: pd.DataFrame, target: pd.DataFrame, components: list) -> np.ndarray:
    pieces = []
    for column, sign, weight in components:
        piece = rank(values(reference, column), values(target, column))
        pieces.append(float(weight) * (piece if float(sign) > 0 else 1.0 - piece))
    return np.sum(pieces, axis=0) / sum(float(item[2]) for item in components)


def encode(reference: pd.DataFrame, target: pd.DataFrame,
           reference_direction: np.ndarray, target_direction: np.ndarray,
           cutoff: float, gate: dict) -> np.ndarray:
    reference_margin = np.abs(reference_direction - cutoff)
    margin = rank(reference_margin, np.abs(target_direction - cutoff))
    reference_margin_rank = rank(reference_margin, reference_margin)
    reference_opportunity = float(gate["sign"]) * values(reference, gate["column"])
    target_opportunity = float(gate["sign"]) * values(target, gate["column"])
    opportunity = rank(reference_opportunity, target_opportunity)
    reference_opportunity_rank = rank(reference_opportunity, reference_opportunity)
    weight = float(gate["margin_weight"])
    reference_confidence = (
        weight * reference_margin_rank + (1.0 - weight) * reference_opportunity_rank
    )
    confidence = weight * margin + (1.0 - weight) * opportunity
    selected = rank(reference_confidence, confidence) >= float(gate["gate_quantile"])
    strength = np.where(selected, 0.86 + 0.14 * margin, 0.84 * margin)
    probability = np.where(
        target_direction >= cutoff, 0.5 + 0.5 * strength, 0.5 - 0.5 * strength
    )
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def high_conf(frame: pd.DataFrame, probability: np.ndarray) -> dict | None:
    selected = np.maximum(probability, 1.0 - probability) >= 0.925
    if selected.sum() < 8:
        return None
    prediction = probability >= 0.5
    y = frame.y.astype(int).to_numpy()
    net = (
        np.where(prediction[selected], 1.0, -1.0)
        * frame.fwd_ret_30m.to_numpy(float)[selected] - 0.002
    )
    return {
        "n": int(selected.sum()), "coverage": float(selected.mean()),
        "accuracy": float((prediction[selected] == y[selected]).mean()),
        "net": float(net.mean()),
    }


def evaluation_part(frame: pd.DataFrame, probability: np.ndarray, market: str,
                    label: str, targets: dict) -> pd.DataFrame:
    part = frame[[
        "event_id", "event_time_utc", "market", "ticker", "y", "fwd_ret_30m"
    ]].copy()
    part["prob"] = probability
    if label == PRIMARY_LABEL:
        part["eval_weight"] = float(targets[market]) / len(part)
    return part


def main() -> None:
    runtime_limits.configure()
    data = app.load_search_frame(app.Config())
    sealed = data.source.isin(SEALED_SOURCES)
    if data.loc[sealed, ["y", "fwd_ret_30m"]].notna().any().any():
        raise RuntimeError(f"{CURRENT_LABEL.upper()} seal outcomes exposed")
    opened = data[data.y.notna()].copy()
    label_status = json.loads(
        (ROOT / "data" / f"{CURRENT_LABEL.upper()}_LABEL_STATUS.json").read_text(encoding="utf-8")
    )
    counts = label_status["new_labeled_counts"]
    targets = {
        "US": float(counts[f"US|US_EXACT_{CURRENT_LABEL.upper()}_SEAL"]),
        "KR": float(counts[f"KR|KR_EXACT_{CURRENT_LABEL.upper()}_SEAL"]),
    }
    frames = {}
    features = {}
    for label, by_market in SETS.items():
        frames[label] = {}
        features[label] = {}
        for market, (reference_sources, target_sources) in by_market.items():
            reference = opened[
                opened.market.eq(market) & opened.source.isin(reference_sources)
            ].sort_values("event_time_utc").reset_index(drop=True)
            target = opened[
                opened.market.eq(market) & opened.source.isin(target_sources)
            ].sort_values("event_time_utc").reset_index(drop=True)
            if reference.empty or target.empty:
                raise RuntimeError(f"empty {label}/{market}")
            frames[label][market] = target
            features[label][market] = (app.model_frame(reference), app.model_frame(target))

    direction_rows = json.loads(
        (ROOT / "cache" / f"{CURRENT_LABEL}_rank_search.json").read_text(encoding="utf-8")
    )["rows"]
    local = {}
    for market in ("US", "KR"):
        policies = []
        seen = set()
        candidates = []
        # Preserve the cross-version leaders, but also admit V31-specific and
        # V30/V31-stable directions.  V31 US is a stable-ticker SEC-form stream
        # and is intentionally distributionally different from the news-heavy
        # V29 stream, so an all-history-only shortlist would discard its signal.
        ranked_directions = []
        ranked_directions.extend(direction_rows[market][:8])
        ranked_directions.extend(sorted(
            direction_rows[market],
            key=lambda row: (
                row["sets"][PRIMARY_LABEL]["balanced_accuracy"]
                + row["sets"][PRIMARY_LABEL]["auc"]
                + row["sets"][PRIMARY_LABEL]["edge"]
            ), reverse=True,
        )[:14])
        ranked_directions.extend(sorted(
            direction_rows[market],
            key=lambda row: min(
                row["sets"][label]["balanced_accuracy"]
                + row["sets"][label]["auc"]
                + row["sets"][label]["edge"]
                for label in RECENT
            ), reverse=True,
        )[:10])
        for row in ranked_directions:
            key = json.dumps(row["components"], sort_keys=True) + f"|{row['cutoff']:.4f}"
            if key not in seen:
                candidates.append(row)
                seen.add(key)
            if len(candidates) == 32:
                break
        print("DIRECTION CANDIDATES", market, len(candidates), flush=True)
        for direction_index, candidate in enumerate(candidates):
            direction_policies = []
            raw = {}
            reference_raw = {}
            for label in SETS:
                reference, target = features[label][market]
                reference_raw[label] = direction(
                    reference, reference, candidate["components"]
                )
                raw[label] = direction(reference, target, candidate["components"])
            for column, sign, margin_weight, quantile in itertools.product(
                    GATES, (-1.0, 1.0), (0.0, 0.25, 0.5, 0.75),
                    ((0.55,) if market=="US" else (0.85,))):
                gate = {
                    "column": column, "sign": sign,
                    "margin_weight": margin_weight, "gate_quantile": quantile,
                }
                probabilities = {}
                summaries = {}
                for label in SETS:
                    reference, target = features[label][market]
                    probability = encode(
                        reference, target, reference_raw[label], raw[label],
                        float(candidate["cutoff"]), gate,
                    )
                    probabilities[label] = probability
                    summaries[label] = high_conf(frames[label][market], probability)
                if any(value is None for value in summaries.values()):
                    continue
                current = summaries[PRIMARY_LABEL]
                checks = sum((current["accuracy"] >= 0.60,
                              current["net"] > 0,
                              current["coverage"] >= 0.15))
                checks += sum(summaries[label]["accuracy"] >= 0.55 for label in OLDER)
                checks += sum(summaries[label]["net"] > -0.002 for label in OLDER)
                checks += sum(summaries[label]["accuracy"] >= 0.55 for label in RECENT)
                checks += sum(summaries[label]["net"] > -0.002 for label in RECENT)
                score = (
                    4.0 * current["accuracy"] + 80.0 * current["net"]
                    + current["coverage"]
                    + 0.75 * summaries["v32"]["accuracy"]
                    + 15.0 * summaries["v32"]["net"]
                    + 0.25 * summaries["v31"]["accuracy"]
                    + 5.0 * summaries["v31"]["net"]
                )
                direction_policies.append({
                    "market": market, "direction_index": direction_index,
                    "direction": candidate, "gate": gate,
                    "summaries": summaries, "checks": int(checks),
                    "score": float(score), "probabilities": probabilities,
                })
            direction_policies.sort(
                key=lambda item: (item["checks"], item["score"]), reverse=True
            )
            policies.extend(direction_policies[:10])
            if (direction_index + 1) % 10 == 0:
                print("GATES", market, direction_index + 1, "/", len(candidates), flush=True)
        policies.sort(key=lambda item: (item["checks"], item["score"]), reverse=True)
        local[market] = policies[:40]
        for item in local[market][:10]:
            print("LOCAL", market, json.dumps({
                key: value for key, value in item.items() if key != "probabilities"
            }, ensure_ascii=False), flush=True)

    combined = []
    for us, kr in itertools.product(local["US"], local["KR"]):
        result = {}
        checks = []
        for label in SETS:
            parts = [
                evaluation_part(frames[label]["US"], us["probabilities"][label], "US", label, targets),
                evaluation_part(frames[label]["KR"], kr["probabilities"][label], "KR", label, targets),
            ]
            metrics = app.evaluate(pd.concat(parts, ignore_index=True), 0.925, 0.002)
            result[label] = {"metrics": metrics}
            if label == PRIMARY_LABEL:
                checks.extend((
                    metrics["balanced_accuracy"] >= 0.54,
                    metrics["auc"] >= 0.56,
                    metrics["edge_vs_naive"] >= 0.02,
                    metrics["highconf_coverage"] >= 0.15,
                    metrics["highconf_accuracy"] >= 0.60,
                    metrics["strategy_mean_signed_net"] > 0,
                    metrics["by_market"]["US"]["balanced_accuracy"] >= 0.51,
                    metrics["by_market"]["KR"]["balanced_accuracy"] >= 0.51,
                ))
        current = result[PRIMARY_LABEL]["metrics"]
        score = (
            current["balanced_accuracy"] + current["auc"]
            + current["edge_vs_naive"] + 2.0 * current["highconf_accuracy"]
            + 50.0 * current["strategy_mean_signed_net"]
            + 0.15 * min(result[label]["metrics"]["balanced_accuracy"]
                         for label in ("v31", "v32"))
            + 0.15 * min(result[label]["metrics"]["auc"]
                         for label in ("v31", "v32"))
        )
        combined.append({
            "policies": {
                "US": {key: value for key, value in us.items() if key != "probabilities"},
                "KR": {key: value for key, value in kr.items() if key != "probabilities"},
            },
            "sets": result, "checks": int(sum(checks)), "score": float(score),
        })
    combined.sort(key=lambda item: (item["checks"], item["score"]), reverse=True)

    # Reconstruct exact probability arrays only for finalists, then clustered bootstrap.
    for row in combined[:10]:
        for label in (PRIMARY_LABEL,):
            parts = []
            for market in ("US", "KR"):
                policy = row["policies"][market]
                reference, target = features[label][market]
                reference_direction = direction(
                    reference, reference, policy["direction"]["components"]
                )
                target_direction = direction(
                    reference, target, policy["direction"]["components"]
                )
                probability = encode(
                    reference, target, reference_direction, target_direction,
                    float(policy["direction"]["cutoff"]), policy["gate"],
                )
                parts.append(evaluation_part(
                    frames[label][market], probability, market, label, targets
                ))
            row["sets"][label]["bootstrap"] = app.bootstrap_by_day(
                pd.concat(parts, ignore_index=True), 0.925, 0.002, nboot=1000
            )
        row["checks"] += sum(
            row["sets"][label]["bootstrap"]["balanced_accuracy_lower95"] >= 0.50
            for label in (PRIMARY_LABEL,)
        )
    combined[:10] = sorted(
        combined[:10], key=lambda item: (item["checks"], item["score"]), reverse=True
    )
    path = ROOT / "cache" / f"{CURRENT_LABEL}_confidence.json"
    path.write_text(json.dumps({
        f"{CURRENT_LABEL}_seal_outcomes_loaded": False, "runtime": runtime_limits.status(),
        "target_market_counts": targets, "policies": combined[:200],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in combined[:20]:
        print("FINAL", json.dumps(row, ensure_ascii=False), flush=True)
    print("RESULT=" + str(path), flush=True)


if __name__ == "__main__":
    main()
