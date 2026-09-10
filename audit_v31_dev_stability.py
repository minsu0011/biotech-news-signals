"""Label-safe V31 DEV hash-half stability audit for frozen-policy finalists."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import bio_news_30m_v3 as app
from experiment_v28_confidence import direction, encode, evaluation_part


ROOT = Path(__file__).resolve().parent


def bucket(market: str, event_id: str) -> int:
    value = str(event_id)
    if market == "KR":
        value = value.rsplit(":", 1)[-1]
    return int(hashlib.sha256(f"V31|{market}|{value}".encode()).hexdigest(), 16) % 10


def main() -> None:
    cfg = app.Config()
    data = app.load_search_frame(cfg)
    seal = data.source.isin({"US_EXACT_V31_SEAL", "KR_EXACT_V31_SEAL"})
    if data.loc[seal, ["y", "fwd_ret_30m"]].notna().any().any():
        raise RuntimeError("V31 seal outcomes exposed")
    opened = data[data.y.notna()].copy()
    counts = json.loads(
        (ROOT / "data" / "V31_LABEL_STATUS.json").read_text(encoding="utf-8")
    )["new_labeled_counts"]
    targets = {
        "US": float(counts["US|US_EXACT_V31_SEAL"]),
        "KR": float(counts["KR|KR_EXACT_V31_SEAL"]),
    }
    frames = {}
    features = {}
    for market in ("US", "KR"):
        frame = opened[
            opened.market.eq(market) & opened.source.eq(f"{market}_EXACT_V31_DEV")
        ].sort_values("event_time_utc").reset_index(drop=True)
        frames[market] = frame
        features[market] = app.model_frame(frame)

    finalists = json.loads(
        (ROOT / "cache" / "v31_confidence.json").read_text(encoding="utf-8")
    )["policies"]
    results = []
    for index, finalist in enumerate(finalists):
        parts = []
        for market in ("US", "KR"):
            policy = finalist["policies"][market]
            feature = features[market]
            raw = direction(feature, feature, policy["direction"]["components"])
            probability = encode(
                feature, feature, raw, raw,
                float(policy["direction"]["cutoff"]), policy["gate"],
            )
            part = evaluation_part(frames[market], probability, market, "v31", targets)
            part["hash_bucket"] = [bucket(market, value) for value in part.event_id]
            parts.append(part)
        combined = pd.concat(parts, ignore_index=True)
        summaries = {}
        for label, mask in (
            ("bucket_0_1", combined.hash_bucket.lt(2)),
            ("bucket_2_3", combined.hash_bucket.ge(2)),
        ):
            part = combined.loc[mask].copy()
            for market in ("US", "KR"):
                selected = part.market.eq(market)
                part.loc[selected, "eval_weight"] = targets[market] / int(selected.sum())
            summaries[label] = app.evaluate(part, 0.925, cfg.round_trip_cost)
        values = list(summaries.values())
        checks = sum(
            metric["balanced_accuracy"] >= 0.53
            and metric["auc"] >= 0.55
            and metric["edge_vs_naive"] >= 0.01
            and metric["highconf_coverage"] >= 0.12
            and metric["highconf_accuracy"] >= 0.58
            and metric["strategy_mean_signed_net"] > -0.001
            and metric["by_market"]["US"]["balanced_accuracy"] >= 0.50
            and metric["by_market"]["KR"]["balanced_accuracy"] >= 0.50
            for metric in values
        )
        score = (
            min(metric["balanced_accuracy"] for metric in values)
            + min(metric["auc"] for metric in values)
            + min(metric["edge_vs_naive"] for metric in values)
            + 1.5 * min(metric["highconf_accuracy"] for metric in values)
            + 40.0 * min(metric["strategy_mean_signed_net"] for metric in values)
        )
        results.append({
            "finalist_index": index, "half_pass_count": checks,
            "score": float(score), "halves": summaries,
            "full_v31": finalist["sets"]["v31"],
            "policies": finalist["policies"],
        })
    results.sort(key=lambda row: (row["half_pass_count"], row["score"]), reverse=True)
    output = {
        "status": "PASS" if results[0]["half_pass_count"] == 2 else "REVIEW",
        "v31_seal_outcomes_loaded": False,
        "rows": results[:50],
    }
    path = ROOT / "output" / "V31" / "DEV_HASH_HALF_STABILITY.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results[:15]:
        print("TOP", row["finalist_index"], row["half_pass_count"], round(row["score"], 6))
        for label, metric in row["halves"].items():
            print(label, {key: round(metric[key], 6) for key in (
                "balanced_accuracy", "auc", "edge_vs_naive", "highconf_coverage",
                "highconf_accuracy", "strategy_mean_signed_net",
            )}, {market: round(metric["by_market"][market]["balanced_accuracy"], 6)
                for market in ("US", "KR")})
    print("RESULT=" + str(path))


if __name__ == "__main__":
    main()
