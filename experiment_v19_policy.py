"""V19 DEV-only robust rank-policy exploration.

The loader deliberately leaves SEC_V19_SEAL/KIND_V19_SEAL outcomes unparsed.
Previously opened V18 sources may be used as development audits.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import bio_news_30m_v3 as app


def hbucket(value: str, modulo: int) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16) % modulo


def rank(values: pd.Series) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce")
    fill = float(x.median()) if x.notna().any() else 0.0
    return x.fillna(fill).rank(method="average", pct=True).to_numpy(float)


def direction_score(frame: pd.DataFrame, components: tuple[tuple[str, float], ...]) -> np.ndarray:
    scores = [rank(frame[column]) if sign > 0 else 1.0 - rank(frame[column])
              for column, sign in components]
    return np.mean(scores, axis=0)


def metric(frame: pd.DataFrame, score: np.ndarray, cutoff: float = 0.5) -> dict:
    y = frame["y"].astype(int).to_numpy()
    pred = (score >= cutoff).astype(int)
    signed = np.where(pred == 1, 1.0, -1.0) * frame["fwd_ret_30m"].to_numpy(float)
    return {
        "n": len(frame),
        "up": float(y.mean()),
        "bal": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, score)),
        "acc": float((pred == y).mean()),
        "net": float(signed.mean() - 0.002),
    }


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    event = frame["event_id"].astype(str)
    time = pd.to_datetime(frame["event_time_utc"], utc=True)
    order = time.rank(method="first", pct=True).to_numpy(float)
    hb2 = event.map(lambda x: hbucket(x, 2)).to_numpy()
    hb3 = event.map(lambda x: hbucket(x, 3)).to_numpy()
    out = {"all": np.ones(len(frame), dtype=bool)}
    out.update({f"h2_{i}": hb2 == i for i in range(2)})
    out.update({f"h3_{i}": hb3 == i for i in range(3)})
    out.update({f"t3_{i}": (order > i / 3) & (order <= (i + 1) / 3) for i in range(3)})
    return out


def robust_grid(frame: pd.DataFrame, candidates: list[tuple[tuple[str, float], ...]], top: int = 20) -> list[dict]:
    audit_masks = masks(frame)
    rows = []
    for components in candidates:
        score = direction_score(frame, components)
        for cutoff in np.arange(0.40, 0.601, 0.025):
            audits = {name: metric(frame.loc[mask], score[mask], float(cutoff))
                      for name, mask in audit_masks.items() if mask.sum() >= 20}
            core = [audits[name] for name in audits if name == "all" or name.startswith("h2_")]
            value = (min(x["auc"] for x in core) + min(x["bal"] for x in core)
                     + audits["all"]["auc"] + audits["all"]["bal"])
            rows.append({
                "components": components,
                "cutoff": round(float(cutoff), 3),
                "value": float(value),
                "min_core_auc": min(x["auc"] for x in core),
                "min_core_bal": min(x["bal"] for x in core),
                "audits": audits,
            })
    return sorted(rows, key=lambda x: x["value"], reverse=True)[:top]


def kr_univariate(parts: dict[str, pd.DataFrame], top: int = 40) -> list[dict]:
    rows = []
    for column in app.NUMERIC_COLS:
        if any(column not in frame for frame in parts.values()):
            continue
        for sign in (-1.0, 1.0):
            audits = {}
            for name, frame in parts.items():
                score = rank(frame[column]) if sign > 0 else 1.0 - rank(frame[column])
                best = max((metric(frame, score, float(c)) for c in np.arange(0.35, 0.651, 0.025)),
                           key=lambda x: x["bal"])
                audits[name] = best
            # V19 is a very small audit; insist on directional agreement while
            # ranking primarily on the two much larger, independently opened V18 parts.
            v18_min_auc = min(audits["v18_dev"]["auc"], audits["v18_seal"]["auc"])
            v18_min_bal = min(audits["v18_dev"]["bal"], audits["v18_seal"]["bal"])
            value = v18_min_auc + v18_min_bal + 0.35 * (
                audits["v19_dev"]["auc"] + audits["v19_dev"]["bal"]
            )
            rows.append({"column": column, "sign": sign, "value": value,
                         "v18_min_auc": v18_min_auc, "v18_min_bal": v18_min_bal,
                         "audits": audits})
    return sorted(rows, key=lambda x: x["value"], reverse=True)[:top]


def kr_robust_grid(parts: dict[str, pd.DataFrame], top: int = 30) -> list[dict]:
    atoms = (
        ("pre_ret_120", 1.0),
        ("vwap_distance_30", 1.0),
        ("trend_slope_30", 1.0),
        ("excess_ret_15", -1.0),
        ("pre_ret_20", -1.0),
        ("pre_ret_90", 1.0),
        ("headline_len", 1.0),
        ("excess_ret_5", -1.0),
    )
    candidates = []
    for size in (1, 2, 3):
        candidates.extend(tuple(x) for x in itertools.combinations(atoms, size))
    rows = []
    for components in candidates:
        scores = {name: direction_score(frame, components) for name, frame in parts.items()}
        for cutoff in np.arange(0.375, 0.626, 0.025):
            audits = {name: metric(parts[name], scores[name], float(cutoff)) for name in parts}
            v18_min_bal = min(audits["v18_dev"]["bal"], audits["v18_seal"]["bal"])
            v18_min_auc = min(audits["v18_dev"]["auc"], audits["v18_seal"]["auc"])
            min_bal = min(value["bal"] for value in audits.values())
            value = (2.0 * min_bal + v18_min_bal + v18_min_auc
                     + 0.25 * audits["v19_dev"]["auc"])
            rows.append({"components": components, "cutoff": round(float(cutoff), 3),
                         "value": value, "min_bal": min_bal,
                         "v18_min_bal": v18_min_bal, "v18_min_auc": v18_min_auc,
                         "audits": audits})
    return sorted(rows, key=lambda x: x["value"], reverse=True)[:top]


def confidence_grid(frame: pd.DataFrame, components: tuple[tuple[str, float], ...],
                    cutoff: float, top: int = 2000) -> list[dict]:
    direction = direction_score(frame, components)
    pred = (direction >= cutoff).astype(int)
    y = frame["y"].astype(int).to_numpy()
    returns = frame["fwd_ret_30m"].to_numpy(float)
    margin = rank(pd.Series(np.abs(direction - cutoff)))
    opportunity_columns = (
        "abs_pre_ret_2", "abs_pre_ret_5", "abs_pre_ret_15", "abs_pre_ret_30",
        "abs_pre_ret_60", "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
        "range_5m", "range_10m", "range_30m", "range_60m",
        "volume_ratio_2_30", "volume_ratio_5_30", "entry_bar_range",
        "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15",
        "excess_ret_5", "excess_ret_15", "excess_ret_30",
    )
    audit_masks = masks(frame)
    rows = []
    for column in opportunity_columns:
        base = pd.to_numeric(frame[column], errors="coerce")
        # Benchmark/excess return magnitudes are opportunities; all other
        # volatility/range/absolute-return fields are already nonnegative.
        if column.startswith(("benchmark_ret_", "excess_ret_")):
            base = base.abs()
        opportunity = rank(base)
        for weight in (0.0, 0.25, 0.50, 0.75, 1.0):
            conf = weight * margin + (1.0 - weight) * opportunity
            for quantile in (0.65, 0.70, 0.75, 0.80, 0.825, 0.85):
                gate = conf >= np.quantile(conf, quantile)
                audits = {}
                for name, mask in audit_masks.items():
                    selected = mask & gate
                    if selected.sum() < 8:
                        continue
                    correct = (pred[selected] == y[selected]).astype(float)
                    signed = np.where(pred[selected] == 1, 1.0, -1.0) * returns[selected]
                    audits[name] = {"n": int(selected.sum()), "coverage": float(selected.sum() / mask.sum()),
                                    "acc": float(correct.mean()), "net": float(signed.mean() - 0.002)}
                if not {"all", "h2_0", "h2_1"}.issubset(audits):
                    continue
                core = [audits["all"], audits["h2_0"], audits["h2_1"]]
                min_acc = min(value["acc"] for value in core)
                min_net = min(value["net"] for value in core)
                value = (3.0 * audits["all"]["acc"] + 50.0 * audits["all"]["net"]
                         + min_acc + 20.0 * min_net + 0.2 * audits["all"]["coverage"])
                rows.append({"column": column, "margin_weight": weight,
                             "gate_quantile": quantile, "value": value,
                             "min_core_acc": min_acc, "min_core_net": min_net,
                             "audits": audits})
    return sorted(rows, key=lambda x: x["value"], reverse=True)[:top]


def encoded_probability(frame: pd.DataFrame, components: tuple[tuple[str, float], ...],
                        cutoff: float, opportunity_column: str | None = None,
                        margin_weight: float = 1.0, gate_quantile: float = 1.1) -> np.ndarray:
    direction = direction_score(frame, components)
    up = direction >= cutoff
    margin = rank(pd.Series(np.abs(direction - cutoff)))
    high = np.zeros(len(frame), dtype=bool)
    if opportunity_column is not None:
        opportunity = rank(pd.to_numeric(frame[opportunity_column], errors="coerce"))
        confidence_score = margin_weight * margin + (1.0 - margin_weight) * opportunity
        high = confidence_score >= np.quantile(confidence_score, gate_quantile)
    confidence = np.where(high, 0.86 + 0.14 * margin, 0.84 * margin)
    return np.where(up, 0.50 + 0.50 * confidence, 0.50 - 0.50 * confidence)


def combined_dev_metric(us: pd.DataFrame, kr: pd.DataFrame,
                        us_components: tuple[tuple[str, float], ...], us_cutoff: float,
                        kr_components: tuple[tuple[str, float], ...], kr_cutoff: float) -> dict:
    parts = []
    for market, frame, probability, target in (
        ("US", us, encoded_probability(us, us_components, us_cutoff, "range_5m", 0.25, 0.80), 832.0),
        ("KR", kr, encoded_probability(kr, kr_components, kr_cutoff), 102.0),
    ):
        part = frame[["event_id", "event_time_utc", "y", "fwd_ret_30m"]].copy()
        part["market"] = market
        part["prob"] = probability
        part["eval_weight"] = target / len(part)
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    metrics = app.evaluate(combined, 0.925, 0.002)
    metrics["bootstrap"] = app.bootstrap_by_day(combined, 0.925, 0.002, nboot=600)
    return metrics


def main() -> None:
    cfg = app.Config()
    data = app.load_search_frame(cfg)
    assert data.loc[data.source.isin({"SEC_V19_SEAL", "KIND_V19_SEAL"}), "y"].isna().all()

    us_raw = data[data.source.eq("SEC_V19_DEV") & data.y.notna()].copy().reset_index(drop=True)
    us = app.model_frame(us_raw)
    us[["y", "fwd_ret_30m"]] = us_raw[["y", "fwd_ret_30m"]]
    kr_parts = {
        "v18_dev": data[data.source.eq("KIND_V18_DEV") & data.y.notna()].copy().reset_index(drop=True),
        "v18_seal": data[data.source.eq("KIND_V18_SEAL") & data.y.notna()].copy().reset_index(drop=True),
        "v19_dev": data[data.source.eq("KIND_V19_DEV") & data.y.notna()].copy().reset_index(drop=True),
    }
    for name, raw in list(kr_parts.items()):
        features = app.model_frame(raw)
        features[["y", "fwd_ret_30m"]] = raw[["y", "fwd_ret_30m"]]
        kr_parts[name] = features

    us_atoms = (
        ("benchmark_ret_5", -1.0),
        ("benchmark_ret_2", -1.0),
        ("close_position_60", -1.0),
        ("close_position_10", -1.0),
        ("pre_ret_45", -1.0),
        ("excess_ret_15", -1.0),
    )
    us_candidates = []
    for size in (1, 2, 3):
        us_candidates.extend(tuple(x) for x in itertools.combinations(us_atoms, size))

    us_grid = robust_grid(us, us_candidates)
    chosen_us = tuple(tuple(value) for value in us_grid[0]["components"])
    kr_grid = kr_robust_grid(kr_parts)
    chosen_kr = tuple(tuple(value) for value in kr_grid[0]["components"])
    result = {
        "seal_outcomes_loaded": False,
        "counts": {"us_v19_dev": len(us), **{key: len(value) for key, value in kr_parts.items()}},
        "us_grid": us_grid,
        "us_confidence_grid": confidence_grid(us, chosen_us, us_grid[0]["cutoff"]),
        "kr_univariate": kr_univariate(kr_parts),
        "kr_grid": kr_grid,
        "combined_dev": combined_dev_metric(
            us, kr_parts["v19_dev"], chosen_us, us_grid[0]["cutoff"],
            chosen_kr, kr_grid[0]["cutoff"],
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
