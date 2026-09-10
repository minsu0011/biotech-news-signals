"""Combine selected repeated V14 DEV OOF directions and inspect confidence."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / len(reference)


def main() -> None:
    raw = pd.read_parquet(app.CACHE / "v14_repeated_oof.parquet")
    us = raw[raw.market.eq("US") & raw.augment_v13 & raw.model.eq("blend")].copy()
    kr = raw[raw.market.eq("KR") & ~raw.augment_v13 & raw.model.eq("cat3")].copy()
    cutoffs = {"US": .485, "KR": .545}
    for part, market, target in ((us, "US", 305.0), (kr, "KR", 1046.0)):
        raw_probability = part.prob.to_numpy(float)
        shifted = app.shift_probability(raw_probability, cutoffs[market])
        direction = shifted >= .5
        distance = np.abs(raw_probability - cutoffs[market])
        confidence = percentile(distance, distance)
        part["prob_shifted"] = shifted
        part["confidence_rank"] = confidence
        part["correct"] = direction == part.y.to_numpy(int)
        part["eval_weight"] = target / len(part)
        print("MARKET", market)
        for q in (.50, .60, .70, .75, .80, .85, .90):
            mask = confidence >= q
            signed = np.where(direction[mask], 1.0, -1.0) * part.loc[mask, "fwd_ret_30m"].to_numpy(float) - .002
            print(q, int(mask.sum()), float(part.loc[mask, "correct"].mean()), float(signed.mean()))
    combined = pd.concat([us, kr], ignore_index=True)
    for blend in (0.0, .25, .5, .75, 1.0):
        direction_probability = combined.prob_shifted.to_numpy(float)
        directional_confidence = 2.0 * np.abs(direction_probability - .5)
        rank_confidence = combined.confidence_rank.to_numpy(float)
        confidence = blend * directional_confidence + (1.0 - blend) * rank_confidence
        probability = np.where(
            direction_probability >= .5,
            .5 + .5 * confidence,
            .5 - .5 * confidence,
        )
        candidate = combined.copy()
        candidate["prob"] = probability
        print("\nCONFIDENCE_BLEND", blend)
        for threshold in np.arange(.55, .951, .025):
            metrics = app.evaluate(candidate, float(threshold), .002)
            if .13 <= metrics["highconf_coverage"] <= .40:
                print(json.dumps({
                    "threshold": float(threshold),
                    "balanced_accuracy": metrics["balanced_accuracy"], "auc": metrics["auc"],
                    "accuracy": metrics["accuracy"], "edge": metrics["edge_vs_naive"],
                    "coverage": metrics["highconf_coverage"], "hc_acc": metrics["highconf_accuracy"],
                    "hc_net": metrics["strategy_mean_signed_net"],
                    "by_market": metrics["by_market"],
                }))


if __name__ == "__main__":
    main()
