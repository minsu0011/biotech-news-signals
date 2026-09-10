"""V22 OOF ensemble, cutoff, and entry-time confidence-gate scan."""
from __future__ import annotations

import runtime_limits

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
TARGET_WEIGHTS = {"US": 295.0, "KR": 578.0}


ENSEMBLES = {
    "US": {
        "u00": {"m00": 1}, "u02": {"m02": 1}, "u04": {"m04": 1},
        "u07": {"m07": 1}, "u00_02": {"m00": 1, "m02": 1},
        "u00_04": {"m00": 1, "m04": 1},
        "u00_02_07": {"m00": 1, "m02": 1, "m07": 1},
        "u00x2_02_07": {"m00": 2, "m02": 1, "m07": 1},
        "u00_02_04_07": {"m00": 1, "m02": 1, "m04": 1, "m07": 1},
    },
    "KR": {
        "k00": {"m00": 1}, "k01": {"m01": 1}, "k02": {"m02": 1},
        "k03": {"m03": 1}, "k05": {"m05": 1}, "k06": {"m06": 1},
        "k07": {"m07": 1}, "k01_02": {"m01": 1, "m02": 1},
        "k02_03": {"m02": 1, "m03": 1},
        "k02_06": {"m02": 1, "m06": 1},
        "k01_02_03": {"m01": 1, "m02": 1, "m03": 1},
        "k01_02_06": {"m01": 1, "m02": 1, "m06": 1},
        "k01_02_03_06": {"m01": 1, "m02": 1, "m03": 1, "m06": 1},
        "k02x2_01_03_06": {"m02": 2, "m01": 1, "m03": 1, "m06": 1},
    },
}


def midrank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    event_hash = frame.event_id.astype(str).map(
        lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % 2,
    ).to_numpy()
    ticker_hash = frame.ticker.astype(str).map(
        lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % 2,
    ).to_numpy()
    time_rank = pd.to_datetime(frame.event_time_utc, utc=True).rank(method="first", pct=True).to_numpy()
    return {
        "all": np.ones(len(frame), dtype=bool),
        "h0": event_hash == 0, "h1": event_hash == 1,
        "t0": time_rank <= .5, "t1": time_rank > .5,
        "ticker0": ticker_hash == 0, "ticker1": ticker_hash == 1,
    }


def direction_audit(frame: pd.DataFrame, probability: np.ndarray, cutoff: float,
                    split_masks: dict[str, np.ndarray], aucs: dict[str, float]) -> dict:
    y = frame.y.astype(int).to_numpy()
    pred = probability >= cutoff
    result = {}
    for name, use in split_masks.items():
        yy = y[use]
        pp = probability[use]
        pr = pred[use]
        accuracy = float((pr == yy).mean())
        up_rate = float(yy.mean())
        positive = yy == 1
        negative = ~positive
        balanced = .5 * (float(pr[positive].mean()) + float((~pr[negative]).mean()))
        result[name] = {
            "n": int(use.sum()), "accuracy": accuracy,
            "balanced_accuracy": balanced, "auc": aucs[name],
            "edge": accuracy - max(up_rate, 1.0 - up_rate),
            "up_prediction_rate": float(pr.mean()),
        }
    return result


def ensemble_probability(wide: pd.DataFrame, weights: dict[str, int]) -> np.ndarray:
    return np.average(
        np.stack([wide[name].to_numpy(float) for name in weights]),
        axis=0, weights=list(weights.values()),
    )


def direction_options(frame: pd.DataFrame, probabilities: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    split_masks = masks(frame)
    y = frame.y.astype(int).to_numpy()
    for ensemble_name, probability in probabilities.items():
        aucs = {
            name: float(roc_auc_score(y[use], probability[use]))
            for name, use in split_masks.items()
        }
        for cutoff in np.arange(.45, .751, .005):
            audit = direction_audit(
                frame, probability, float(cutoff), split_masks, aucs,
            )
            core = list(audit.values())
            full = audit["all"]
            pass_count = int(full["balanced_accuracy"] >= .54) + int(full["edge"] >= .02)
            pass_count += sum(value["balanced_accuracy"] >= .50 for value in core[1:])
            score = (
                10 * full["balanced_accuracy"] + 5 * full["edge"]
                + 3 * min(value["balanced_accuracy"] for value in core)
                + min(value["edge"] for value in core)
            )
            rows.append({
                "ensemble": ensemble_name, "cutoff": float(cutoff),
                "pass_count": pass_count, "score": float(score), "audit": audit,
            })
    return sorted(rows, key=lambda row: (row["pass_count"], row["score"]), reverse=True)


GATE_COLUMNS = (
    "volume_ratio_10_60", "volume_ratio_5_30", "volume_ratio_2_30",
    "pre_vol_10", "pre_vol_30", "abs_pre_ret_5", "abs_pre_ret_15",
    "abs_pre_ret_30", "range_10m", "range_30m", "entry_bar_range",
    "benchmark_ret_5", "benchmark_ret_30", "minutes_to_close",
)


def encode(frame: pd.DataFrame, raw: np.ndarray, cutoff: float, gate: dict,
           feature: pd.DataFrame | None = None) -> np.ndarray:
    if feature is None:
        feature = app.model_frame(frame)
    margin = midrank(np.abs(raw - cutoff))
    opportunity_raw = pd.to_numeric(feature[gate["column"]], errors="coerce")
    fill = float(opportunity_raw.median()) if opportunity_raw.notna().any() else 0.0
    opportunity = midrank(opportunity_raw.fillna(fill).to_numpy(float))
    if gate["sign"] < 0:
        opportunity = 1.0 - opportunity
    confidence_raw = gate["margin_weight"] * margin + (1.0 - gate["margin_weight"]) * opportunity
    confidence_rank = midrank(confidence_raw)
    selected = confidence_rank >= gate["gate_quantile"]
    confidence = np.where(selected, .86 + .14 * margin, .84 * margin)
    return np.where(raw >= cutoff, .5 + .5 * confidence, .5 - .5 * confidence)


def gate_candidates() -> list[dict]:
    return [
        {"column": column, "sign": sign, "margin_weight": weight, "gate_quantile": quantile}
        for column, sign, weight, quantile in itertools.product(
            GATE_COLUMNS, (-1, 1), (.25, .50, .75, 1.0), (.70, .75, .80, .825, .85),
        )
    ]


def combined_frame(parts: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> pd.DataFrame:
    out = []
    for market, (frame, probability) in parts.items():
        z = frame[["event_id", "event_time_utc", "market", "ticker", "y", "fwd_ret_30m"]].copy()
        z["prob"] = probability
        z["eval_weight"] = TARGET_WEIGHTS[market] / len(z)
        out.append(z)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    data = app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V22_SEAL", "NAVER_NEWS_V22_SEAL"}), "y"].notna().any():
        raise RuntimeError("V22 seal outcome exposure detected")
    oof = pd.read_parquet(ROOT / "cache" / "v22_oof_predictions.parquet")
    frames = {
        "US": data[data.source.eq("SEC_V22_DEV") & data.y.notna()].copy().reset_index(drop=True),
        "KR": data[data.source.eq("NAVER_NEWS_V22_DEV") & data.y.notna()].copy().reset_index(drop=True),
    }
    features = {market: app.model_frame(frame) for market, frame in frames.items()}
    probabilities = {}
    options = {}
    for market, frame in frames.items():
        market_oof = oof[oof.market.eq(market)]
        wide = market_oof.pivot(index="event_id", columns="model_name", values="probability")
        wide = wide.reindex(frame.event_id)
        if wide.isna().any().any():
            raise RuntimeError(f"OOF alignment failed: {market}")
        probabilities[market] = {
            name: ensemble_probability(wide, weights)
            for name, weights in ENSEMBLES[market].items()
        }
        options[market] = direction_options(frame, probabilities[market])
        print(f"[{market} direction top]")
        for row in options[market][:12]:
            full = row["audit"]["all"]
            print(
                f" {row['ensemble']} cut={row['cutoff']:.3f} "
                f"bal={full['balanced_accuracy']:.4f} auc={full['auc']:.4f} "
                f"edge={full['edge']:.4f} minbal="
                f"{min(value['balanced_accuracy'] for value in row['audit'].values()):.4f}",
            )

    # Deduplicate near-identical top direction choices, retaining the robust leaders.
    direction_short = {}
    for market in ("US", "KR"):
        seen = set()
        direction_short[market] = []
        for row in options[market]:
            key = (row["ensemble"], round(row["cutoff"], 3))
            if key in seen:
                continue
            if all(abs(row["cutoff"] - old["cutoff"]) >= .01 or row["ensemble"] != old["ensemble"]
                   for old in direction_short[market]):
                direction_short[market].append(row)
                seen.add(key)
            if len(direction_short[market]) >= 10:
                break

    # Rank gates within each market/direction using full and split stability.
    gate_short = {"US": [], "KR": []}
    for market in ("US", "KR"):
        frame = frames[market]
        y = frame.y.astype(int).to_numpy()
        ret = frame.fwd_ret_30m.to_numpy(float)
        split_masks = masks(frame)
        for direction in direction_short[market]:
            raw = probabilities[market][direction["ensemble"]]
            pred = raw >= direction["cutoff"]
            candidates = []
            for gate in gate_candidates():
                encoded = encode(frame, raw, direction["cutoff"], gate, features[market])
                selected = np.maximum(encoded, 1 - encoded) >= .925
                split = {}
                for split_name, split_mask in split_masks.items():
                    use = selected & split_mask
                    if use.sum() < 15:
                        break
                    signed_net = np.where(pred[use], 1.0, -1.0) * ret[use] - .002
                    split[split_name] = {
                        "n": int(use.sum()), "coverage": float(use.sum() / split_mask.sum()),
                        "accuracy": float((pred[use] == y[use]).mean()),
                        "net": float(signed_net.mean()),
                    }
                if len(split) != len(split_masks):
                    continue
                full = split["all"]
                score = (
                    5 * full["accuracy"] + 60 * full["net"] + full["coverage"]
                    + min(value["accuracy"] for value in split.values())
                    + 20 * min(value["net"] for value in split.values())
                )
                candidates.append({"gate": gate, "score": float(score), "split": split})
            candidates.sort(key=lambda row: row["score"], reverse=True)
            for candidate in candidates[:6]:
                gate_short[market].append({
                    "direction": {key: direction[key] for key in ("ensemble", "cutoff")},
                    **candidate,
                })
        gate_short[market].sort(key=lambda row: row["score"], reverse=True)
        gate_short[market] = gate_short[market][:30]
        print(f"[{market} gate top]")
        for row in gate_short[market][:8]:
            full = row["split"]["all"]
            print(
                f" {row['direction']} {row['gate']} "
                f"hc_acc={full['accuracy']:.4f} cov={full['coverage']:.4f} net={full['net']:.5f}",
            )

    pairs = []
    for us, kr in itertools.product(gate_short["US"], gate_short["KR"]):
        encoded = {}
        for market, choice in (("US", us), ("KR", kr)):
            direction = choice["direction"]
            encoded[market] = encode(
                frames[market], probabilities[market][direction["ensemble"]],
                direction["cutoff"], choice["gate"], features[market],
            )
        combined = combined_frame({market: (frames[market], encoded[market]) for market in ("US", "KR")})
        metric = app.evaluate(combined, .925, .002)
        required = [
            metric["balanced_accuracy"] >= .54, metric["auc"] >= .56,
            metric["edge_vs_naive"] >= .02, metric["highconf_coverage"] >= .15,
            metric["highconf_accuracy"] >= .60, metric["strategy_mean_signed_net"] > 0,
            metric["by_market"]["US"]["balanced_accuracy"] >= .51,
            metric["by_market"]["KR"]["balanced_accuracy"] >= .51,
        ]
        score = (
            4 * metric["balanced_accuracy"] + 2 * metric["auc"]
            + 3 * metric["highconf_accuracy"] + 40 * metric["strategy_mean_signed_net"]
            + metric["edge_vs_naive"] + .2 * metric["highconf_coverage"]
        )
        pairs.append({
            "pass_count": int(sum(required)), "score": float(score),
            "us": us, "kr": kr, "metrics": metric,
        })
    pairs.sort(key=lambda row: (row["pass_count"], row["score"]), reverse=True)
    # Bootstrap only finalists; it cannot influence the direction/gate search.
    finalists = pairs[:20]
    for row in finalists:
        encoded = {}
        for market, choice in (("US", row["us"]), ("KR", row["kr"])):
            direction = choice["direction"]
            encoded[market] = encode(
                frames[market], probabilities[market][direction["ensemble"]],
                direction["cutoff"], choice["gate"], features[market],
            )
        combined = combined_frame({market: (frames[market], encoded[market]) for market in ("US", "KR")})
        row["bootstrap"] = app.bootstrap_by_day(combined, .925, .002, nboot=200)
        row["pass_count"] += int(row["bootstrap"]["balanced_accuracy_lower95"] >= .50)
    finalists.sort(key=lambda row: (row["pass_count"], row["score"]), reverse=True)
    result = {
        "seal_outcomes_loaded": False, "runtime": runtime_limits.status(),
        "direction_top": {market: rows[:30] for market, rows in options.items()},
        "gate_top": gate_short, "pairs": finalists,
    }
    (ROOT / "cache" / "v22_ensemble_audits.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("[PAIR TOP]")
    for row in finalists[:10]:
        metric = row["metrics"]
        print(
            f"passes={row['pass_count']}/9 bal={metric['balanced_accuracy']:.4f} "
            f"auc={metric['auc']:.4f} edge={metric['edge_vs_naive']:.4f} "
            f"hc={metric['highconf_accuracy']:.4f}/{metric['highconf_coverage']:.4f} "
            f"net={metric['strategy_mean_signed_net']:.5f} "
            f"boot={row['bootstrap']['balanced_accuracy_lower95']:.4f}",
        )


if __name__ == "__main__":
    main()
