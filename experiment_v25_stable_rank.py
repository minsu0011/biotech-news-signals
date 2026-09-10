"""V25 label-safe cross-version search for small monotone rank policies.

V25 SEAL outcomes are never loaded.  Candidate signs, weights and direction
cutoffs must work on both the already-open V24 SEAL audit and V25 DEV.  The
V24 audit uses V24 DEV feature distributions as its frozen rank reference;
V25 DEV uses its own label-independent feature distribution, matching the
reference that will be frozen before V25 SEAL is opened.
"""
from __future__ import annotations

import itertools
import json
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import runtime_limits
import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
TARGET = {"US": 295.0, "KR": 521.0}
SOURCE = {
    "US": {"audit_ref": "SEC_V24_DEV", "audit": "SEC_V24_SEAL", "dev": "SEC_V25_DEV"},
    "KR": {"audit_ref": "NAVER_NEWS_V24_DEV", "audit": "NAVER_NEWS_V24_SEAL",
           "dev": "NAVER_NEWS_V25_DEV"},
}

# Deliberately exclude identifiers, source, ticker and text.  These are
# entry-time numerical state variables only.
FEATURES = tuple(app.NUMERIC_COLS)


def midrank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)])
    if not len(reference):
        return np.full(len(values), .5)
    fill = float(np.median(reference))
    values = np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right + 1.0) / (2.0 * len(reference))


def matrices(ref: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    xr = app.model_frame(ref)
    xt = app.model_frame(target)
    columns = []
    for column in FEATURES:
        rv = pd.to_numeric(xr[column], errors="coerce").to_numpy(float)
        tv = pd.to_numeric(xt[column], errors="coerce").to_numpy(float)
        columns.append(midrank(rv, tv))
    return np.column_stack(columns)


def metric(y: np.ndarray, score: np.ndarray, cutoff: float) -> dict:
    pred = (score >= cutoff).astype(np.int8)
    up = float(y.mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    pos = int((y == 1).sum()); neg = len(y)-pos
    accuracy = (tp+tn)/len(y)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(.5*(tp/pos+tn/neg)),
        "edge": float(accuracy - max(up, 1.0-up)),
        "pred_up": float(pred.mean()),
    }


def generate_weights(indices: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    # Every univariate policy plus equal-weight pairs/triples are deterministic.
    for size in (1, 2, 3):
        for chosen in itertools.combinations(indices, size):
            w = np.zeros(len(FEATURES)); w[list(chosen)] = 1.0 / size; out.append(w)
    # Random sparse convex blends explore interactions without permitting
    # canceling coefficients or a high-dimensional unconstrained fit.
    for _ in range(6000):
        size = int(rng.integers(2, min(7, len(indices)+1)))
        chosen = rng.choice(indices, size=size, replace=False)
        values = rng.dirichlet(np.full(size, 1.5))
        w = np.zeros(len(FEATURES)); w[chosen] = values; out.append(w)
    return out


def main() -> None:
    data = app.load_search_frame(app.Config())
    seals = {"SEC_V25_SEAL", "NAVER_NEWS_V25_SEAL"}
    if data.loc[data.source.isin(seals), "y"].notna().any():
        raise RuntimeError("V25 seal outcome exposure detected")
    result = {"seal_outcomes_loaded": False, "runtime": runtime_limits.status(),
              "target": TARGET, "markets": {}}
    rng = np.random.default_rng(app.SEED + 25)
    for market in ("US", "KR"):
        src = SOURCE[market]
        frames = {
            name: data[data.source.eq(source) & data.y.notna()].copy()
            .sort_values("event_time_utc").reset_index(drop=True)
            for name, source in src.items()
        }
        # V24 opened-seal audit has its original V24 DEV rank reference.  V25
        # uses its own feature reference, which is label-independent.
        audit_x = matrices(frames["audit_ref"], frames["audit"])
        dev_x = matrices(frames["dev"], frames["dev"])
        audit_y = frames["audit"].y.astype(int).to_numpy()
        dev_y = frames["dev"].y.astype(int).to_numpy()

        univariate = []
        oriented_audit = np.empty_like(audit_x)
        oriented_dev = np.empty_like(dev_x)
        stable = []
        for index, column in enumerate(FEATURES):
            aa = float(roc_auc_score(audit_y, audit_x[:, index]))
            da = float(roc_auc_score(dev_y, dev_x[:, index]))
            # Direction is admitted only when both versions agree.  Orient it
            # using their joint sign, never one split in isolation.
            if aa < .5 and da < .5:
                sign = -1
            elif aa > .5 and da > .5:
                sign = 1
            else:
                sign = 0
            oriented_audit[:, index] = audit_x[:, index] if sign >= 0 else 1-audit_x[:, index]
            oriented_dev[:, index] = dev_x[:, index] if sign >= 0 else 1-dev_x[:, index]
            oa = aa if sign >= 0 else 1-aa
            od = da if sign >= 0 else 1-da
            if sign and min(oa, od) >= .515:
                stable.append(index)
            univariate.append({"column": column, "sign": sign,
                               "audit_auc": aa, "dev_auc": da,
                               "oriented_audit_auc": oa,
                               "oriented_dev_auc": od})

        candidate_heap = []
        serial = 0
        seen = set()
        for weights in generate_weights(stable, rng):
            key = tuple(np.round(weights, 5))
            if key in seen: continue
            seen.add(key)
            audit_score = oriented_audit @ weights
            dev_score = oriented_dev @ weights
            audit_auc = float(roc_auc_score(audit_y, audit_score))
            dev_auc = float(roc_auc_score(dev_y, dev_score))
            if min(audit_auc, dev_auc) < .525: continue
            components = [{"column": FEATURES[i],
                           "sign": int(univariate[i]["sign"]),
                           "weight": float(weights[i])}
                          for i in np.flatnonzero(weights)]
            for cutoff in np.arange(.50, .951, .01):
                am = metric(audit_y, audit_score, float(cutoff))
                dm = metric(dev_y, dev_score, float(cutoff))
                am["auc"] = audit_auc; dm["auc"] = dev_auc
                # Reward cross-version lower bounds and a low DEV/audit gap.
                floor_bal = min(am["balanced_accuracy"], dm["balanced_accuracy"])
                floor_edge = min(am["edge"], dm["edge"])
                objective = (8*floor_bal + 5*floor_edge +
                             2*min(audit_auc, dev_auc) -
                             abs(am["balanced_accuracy"]-dm["balanced_accuracy"]))
                row = {"components": components, "cutoff": float(cutoff),
                       "audit": am, "dev": dm, "objective": float(objective),
                       "robust_passes": int(am["balanced_accuracy"] >= .51) +
                                        int(dm["balanced_accuracy"] >= .51) +
                                        int(am["edge"] >= 0) + int(dm["edge"] >= 0)}
                key = (row["robust_passes"], row["objective"], serial)
                serial += 1
                if len(candidate_heap) < 1500:
                    heapq.heappush(candidate_heap, (key, row))
                elif key > candidate_heap[0][0]:
                    heapq.heapreplace(candidate_heap, (key, row))
        candidates = [item[1] for item in candidate_heap]
        candidates.sort(key=lambda r: (r["robust_passes"], r["objective"]), reverse=True)
        result["markets"][market] = {
            "n": {name: len(frame) for name, frame in frames.items()},
            "up_rate": {name: float(frame.y.mean()) for name, frame in frames.items()},
            "stable_features": [FEATURES[i] for i in stable],
            "univariate": sorted(univariate,
                                 key=lambda r: min(r["oriented_audit_auc"], r["oriented_dev_auc"]),
                                 reverse=True),
            "candidates": candidates[:500],
        }
        print(f"[{market}] stable={len(stable)} retained={len(candidates)} evaluated={serial}", flush=True)
        for row in candidates[:15]:
            names = "+".join(f"{c['sign']:+d}{c['column']}*{c['weight']:.2f}"
                             for c in row["components"])
            print(f"  pass={row['robust_passes']}/4 cut={row['cutoff']:.2f} "
                  f"audit auc/bal/edge={row['audit']['auc']:.3f}/{row['audit']['balanced_accuracy']:.3f}/{row['audit']['edge']:.3f} "
                  f"dev={row['dev']['auc']:.3f}/{row['dev']['balanced_accuracy']:.3f}/{row['dev']['edge']:.3f} {names}", flush=True)
    path = ROOT / "cache" / "v25_stable_rank.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT=" + str(path))


if __name__ == "__main__":
    main()
