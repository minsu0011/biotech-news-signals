"""Cross-audited V27 eligibility and high-confidence policy search.

V27 seal outcomes are never read here.  The two target sets are the already
opened V26 DEV+SEAL audit and V27 DEV.  Direction, eligibility, and confidence
all use entry-time features only.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import runtime_limits


ROOT = Path(__file__).resolve().parent
TARGET = {"US": 258.0, "KR": 325.0}
SETS = {
    "v26": {
        "US": ({"SEC_V26_DEV"}, {"SEC_V26_SEAL"}),
        "KR": ({"NAVER_NEWS_V26_DEV"}, {"NAVER_NEWS_V26_SEAL"}),
    },
    "v27": {
        "US": ({"SEC_V27_DEV"}, {"SEC_V27_DEV"}),
        "KR": ({"NAVER_NEWS_V27_DEV"}, {"NAVER_NEWS_V27_DEV"}),
    },
}
EXCLUDED_GATES = {
    "year", "month", "weekday", "hour", "minute", "minutes_from_open",
    "minutes_to_close", "event_hour", "event_weekday", "event_month",
    "event_hour_sin", "event_hour_cos",
}
GATES = tuple(column for column in app.NUMERIC_COLS if column not in EXCLUDED_GATES)


def rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    fill = float(np.median(reference)) if len(reference) else 0.0
    values = np.nan_to_num(np.asarray(values, float), nan=fill, posinf=fill, neginf=fill)
    if not len(reference):
        return np.full(len(values), 0.5)
    left = np.searchsorted(reference, values, "left")
    right = np.searchsorted(reference, values, "right")
    return (left + right + 1) / (2 * len(reference))


def values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def direction(reference: pd.DataFrame, target: pd.DataFrame, components: list) -> np.ndarray:
    pieces = []
    for column, sign, weight in components:
        component = rank(values(reference, column), values(target, column))
        pieces.append(float(weight) * (component if float(sign) > 0 else 1.0 - component))
    return np.sum(pieces, axis=0) / sum(float(item[2]) for item in components)


def eligibility(reference: pd.DataFrame, target: pd.DataFrame,
                reference_direction: np.ndarray, target_direction: np.ndarray,
                cutoff: float, spec: dict) -> np.ndarray:
    ref_margin = np.abs(reference_direction - cutoff)
    target_margin = np.abs(target_direction - cutoff)
    margin = rank(ref_margin, target_margin)
    ref_margin_rank = rank(ref_margin, ref_margin)
    ref_opportunity = float(spec["sign"]) * values(reference, spec["column"])
    target_opportunity = float(spec["sign"]) * values(target, spec["column"])
    opportunity = rank(ref_opportunity, target_opportunity)
    ref_opportunity_rank = rank(ref_opportunity, ref_opportunity)
    mw = float(spec["margin_weight"])
    reference_confidence = mw * ref_margin_rank + (1.0 - mw) * ref_opportunity_rank
    confidence = mw * margin + (1.0 - mw) * opportunity
    return rank(reference_confidence, confidence) >= float(spec["quantile"])


def encode(reference: pd.DataFrame, target: pd.DataFrame,
           reference_direction: np.ndarray, target_direction: np.ndarray,
           cutoff: float, gate: dict) -> np.ndarray:
    ref_margin = np.abs(reference_direction - cutoff)
    margin = rank(ref_margin, np.abs(target_direction - cutoff))
    ref_margin_rank = rank(ref_margin, ref_margin)
    ref_opportunity = float(gate["sign"]) * values(reference, gate["column"])
    target_opportunity = float(gate["sign"]) * values(target, gate["column"])
    opportunity = rank(ref_opportunity, target_opportunity)
    ref_opportunity_rank = rank(ref_opportunity, ref_opportunity)
    mw = float(gate["margin_weight"])
    reference_confidence = mw * ref_margin_rank + (1.0 - mw) * ref_opportunity_rank
    confidence = mw * margin + (1.0 - mw) * opportunity
    selected = rank(reference_confidence, confidence) >= float(gate["gate_quantile"])
    strength = np.where(selected, 0.86 + 0.14 * margin, 0.84 * margin)
    probability = np.where(target_direction >= cutoff, 0.5 + 0.5 * strength,
                           0.5 - 0.5 * strength)
    return np.clip(probability, 1e-6, 1.0-1e-6)


def evaluation_part(frame: pd.DataFrame, probability: np.ndarray, market: str,
                    label: str) -> pd.DataFrame:
    part = frame[["event_id", "event_time_utc", "market", "ticker", "y", "fwd_ret_30m"]].copy()
    part["prob"] = probability
    # V27 model selection deliberately target-matches the fixed sealed market
    # composition.  V26 remains a literal opened audit with one vote per row.
    if label == "v27":
        part["eval_weight"] = TARGET[market] / len(part)
    return part


def high_conf_summary(frame: pd.DataFrame, probability: np.ndarray) -> dict | None:
    selected = np.maximum(probability, 1.0 - probability) >= 0.925
    if selected.sum() < 5:
        return None
    pred = probability >= 0.5
    y = frame.y.astype(int).to_numpy()
    net = np.where(pred[selected], 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float)[selected] - 0.002
    return {
        "n": int(selected.sum()),
        "coverage": float(selected.mean()),
        "accuracy": float((pred[selected] == y[selected]).mean()),
        "net": float(net.mean()),
    }


def main() -> None:
    runtime_limits.configure()
    data = app.load_search_frame(app.Config())
    sealed = data.source.isin({"SEC_V27_SEAL", "NAVER_NEWS_V27_SEAL"})
    if data.loc[sealed, "y"].notna().any() or data.loc[sealed, "fwd_ret_30m"].notna().any():
        raise RuntimeError("V27 seal outcomes exposed")
    opened = data[data.y.notna()].copy()
    frames: dict = {}
    features: dict = {}
    for label, by_market in SETS.items():
        frames[label] = {}
        features[label] = {}
        for market, (reference_sources, target_sources) in by_market.items():
            reference = opened[opened.market.eq(market) & opened.source.isin(reference_sources)].copy()
            target = opened[opened.market.eq(market) & opened.source.isin(target_sources)].copy()
            reference = reference.sort_values("event_time_utc").reset_index(drop=True)
            target = target.sort_values("event_time_utc").reset_index(drop=True)
            frames[label][market] = target
            features[label][market] = (app.model_frame(reference), app.model_frame(target))

    eligibility_rows = json.loads(
        (ROOT / "cache" / "v27_eligibility.json").read_text(encoding="utf-8")
    )["rows"]
    # Calendar gates and low-sample policies are intentionally excluded.  Keep
    # distinct direction/eligibility policies with at least a 30-event expected
    # cushion over the 500-event certificate minimum.
    candidates = []
    seen = set()
    for source_index, row in enumerate(eligibility_rows):
        e = row["eligibility"]
        if e["column"] in EXCLUDED_GATES or row["expected_v27_n"] < 530 or row["passes"] < 10:
            continue
        key = json.dumps({"markets": row["markets"], "eligibility": e}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidate = dict(row)
        candidate["source_index"] = source_index
        candidates.append(candidate)
        if len(candidates) >= 18:
            break
    if not candidates:
        raise RuntimeError("no robust eligibility candidates")
    print("CANDIDATES", len(candidates), [x["source_index"] for x in candidates], flush=True)

    policies = []
    for candidate_number, candidate in enumerate(candidates):
        raw: dict = {label: {} for label in SETS}
        ref_raw: dict = {label: {} for label in SETS}
        masks: dict = {label: {} for label in SETS}
        filtered_frames: dict = {label: {} for label in SETS}
        filtered_features: dict = {label: {} for label in SETS}
        for label in SETS:
            for market in ("US", "KR"):
                reference, target = features[label][market]
                spec = candidate["markets"][market]
                ref_raw[label][market] = direction(reference, reference, spec["components"])
                raw[label][market] = direction(reference, target, spec["components"])
                mask = np.ones(len(target), dtype=bool)
                if market == "KR":
                    mask = eligibility(reference, target, ref_raw[label][market], raw[label][market],
                                       float(spec["cutoff"]), candidate["eligibility"])
                masks[label][market] = mask
                filtered_frames[label][market] = frames[label][market].loc[mask].reset_index(drop=True)
                filtered_features[label][market] = target.loc[mask].reset_index(drop=True)

        short: dict = {}
        for market in ("US", "KR"):
            rows = []
            spec = candidate["markets"][market]
            for column, sign, mw, quantile in itertools.product(
                    GATES, (-1.0, 1.0), (0.25, 0.5, 0.75), (0.80, 0.825, 0.85)):
                gate = {"column": column, "sign": sign, "margin_weight": mw,
                        "gate_quantile": quantile}
                summaries = {}
                probabilities = {}
                for label in SETS:
                    reference, _ = features[label][market]
                    probability = encode(
                        reference, filtered_features[label][market], ref_raw[label][market],
                        raw[label][market][masks[label][market]], float(spec["cutoff"]), gate,
                    )
                    probabilities[label] = probability
                    summaries[label] = high_conf_summary(filtered_frames[label][market], probability)
                if any(value is None for value in summaries.values()):
                    continue
                robust = sum([
                    summaries["v26"]["accuracy"] >= 0.57,
                    summaries["v27"]["accuracy"] >= 0.57,
                    summaries["v26"]["net"] > 0,
                    summaries["v27"]["net"] > 0,
                ])
                score = (4.0 * min(value["accuracy"] for value in summaries.values()) +
                         80.0 * min(value["net"] for value in summaries.values()) +
                         min(value["coverage"] for value in summaries.values()))
                rows.append({"gate": gate, "summaries": summaries, "robust": robust,
                             "score": float(score), "probabilities": probabilities})
            rows.sort(key=lambda item: (item["robust"], item["score"]), reverse=True)
            short[market] = rows[:50]

        combined = []
        for us_gate, kr_gate in itertools.product(short["US"], short["KR"]):
            sets = {}
            checks = []
            for label in SETS:
                parts = [
                    evaluation_part(filtered_frames[label]["US"], us_gate["probabilities"][label], "US", label),
                    evaluation_part(filtered_frames[label]["KR"], kr_gate["probabilities"][label], "KR", label),
                ]
                metrics = app.evaluate(pd.concat(parts, ignore_index=True), 0.925, 0.002)
                sets[label] = {"metrics": metrics}
                checks.extend([
                    metrics["balanced_accuracy"] >= 0.54,
                    metrics["auc"] >= 0.56,
                    metrics["edge_vs_naive"] >= 0.02,
                    metrics["highconf_coverage"] >= 0.15,
                    metrics["highconf_accuracy"] >= 0.60,
                    metrics["strategy_mean_signed_net"] > 0,
                    metrics["by_market"]["US"]["balanced_accuracy"] >= 0.51,
                    metrics["by_market"]["KR"]["balanced_accuracy"] >= 0.51,
                ])
            a = sets["v26"]["metrics"]
            c = sets["v27"]["metrics"]
            score = (min(a["balanced_accuracy"], c["balanced_accuracy"]) +
                     min(a["auc"], c["auc"]) + min(a["edge_vs_naive"], c["edge_vs_naive"]) +
                     2.0 * min(a["highconf_accuracy"], c["highconf_accuracy"]) +
                     50.0 * min(a["strategy_mean_signed_net"], c["strategy_mean_signed_net"]))
            combined.append({
                "candidate_number": candidate_number,
                "source_index": candidate["source_index"],
                "markets": candidate["markets"],
                "eligibility": candidate["eligibility"],
                "expected_v27_n": candidate["expected_v27_n"],
                "retention": candidate["retention"],
                "gates": {"US": us_gate["gate"], "KR": kr_gate["gate"]},
                "sets": sets,
                "passes": int(sum(checks)),
                "score": float(score),
            })
        combined.sort(key=lambda item: (item["passes"], item["score"]), reverse=True)
        policies.extend(combined[:20])
        best = combined[0]
        print("BEST", candidate_number, candidate["source_index"], best["passes"],
              json.dumps({label: best["sets"][label]["metrics"] for label in SETS},
                         ensure_ascii=False), flush=True)

    policies.sort(key=lambda item: (item["passes"], item["score"]), reverse=True)
    # Exact clustered bootstrap is reserved for the finalists.
    for row in policies[:30]:
        candidate = candidates[row["candidate_number"]]
        for label in SETS:
            parts = []
            for market in ("US", "KR"):
                reference, target = features[label][market]
                spec = candidate["markets"][market]
                rr = direction(reference, reference, spec["components"])
                tr = direction(reference, target, spec["components"])
                mask = np.ones(len(target), dtype=bool)
                if market == "KR":
                    mask = eligibility(reference, target, rr, tr, float(spec["cutoff"]),
                                       candidate["eligibility"])
                probability = encode(reference, target.loc[mask].reset_index(drop=True), rr, tr[mask],
                                     float(spec["cutoff"]), row["gates"][market])
                parts.append(evaluation_part(frames[label][market].loc[mask].reset_index(drop=True),
                                             probability, market, label))
            combined_frame = pd.concat(parts, ignore_index=True)
            row["sets"][label]["bootstrap"] = app.bootstrap_by_day(
                combined_frame, 0.925, 0.002, nboot=2000
            )
        row["passes"] += sum(
            row["sets"][label]["bootstrap"]["balanced_accuracy_lower95"] >= 0.50
            for label in SETS
        )
    policies[:30] = sorted(policies[:30], key=lambda item: (item["passes"], item["score"]),
                           reverse=True)
    serializable = policies[:200]
    result = {
        "v27_seal_outcomes_loaded": False,
        "runtime": runtime_limits.status(),
        "candidate_source_indices": [item["source_index"] for item in candidates],
        "policies": serializable,
    }
    path = ROOT / "cache" / "v27_confidence.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in serializable[:20]:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print("RESULT=" + str(path), flush=True)


if __name__ == "__main__":
    main()
