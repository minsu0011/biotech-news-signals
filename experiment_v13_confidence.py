#!/usr/bin/env python
"""DEV-only high-confidence diagnostics for the fixed V13 direction rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent
SEED = 260825
KR_WEIGHT = 761.0
US_WEIGHT = 350.0
ROUND_TRIP_COST = 0.002


def weighted_top_mask(confidence: np.ndarray, weights: np.ndarray, coverage: float) -> np.ndarray:
    order = np.argsort(-confidence, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    count = max(1, int(np.searchsorted(cumulative, coverage, side="left")) + 1)
    mask = np.zeros(len(confidence), dtype=bool)
    mask[order[:count]] = True
    return mask


def report(name: str, confidence: np.ndarray, frame: pd.DataFrame) -> None:
    confidence = np.asarray(confidence, dtype=float)
    weights = frame.eval_weight.to_numpy(float)
    for coverage in (0.15, 0.20, 0.25, 0.30, 0.40):
        selected = weighted_top_mask(confidence, weights, coverage)
        actual_coverage = weights[selected].sum() / weights.sum()
        accuracy = np.average(frame.correct.to_numpy(float)[selected], weights=weights[selected])
        net = np.average(frame.signed_net.to_numpy(float)[selected], weights=weights[selected])
        print(
            name, f"coverage={actual_coverage:.4f}", f"n={selected.sum()}",
            f"accuracy={accuracy:.6f}", f"net={net:.6f}",
            f"US={int((selected & frame.market.eq('US')).sum())}",
            f"KR={int((selected & frame.market.eq('KR')).sum())}",
        )


def main() -> None:
    component = pd.read_parquet(ROOT / "cache" / "v13_components_oof.parquet")
    allowed = {"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"}
    component = component[component.source.isin(allowed)].copy()
    # Select only DEV rows before retaining outcome columns from the labeled file.
    raw = pd.read_csv(ROOT / "data" / "labeled_events_v13.csv.gz")
    data = raw[raw.source.isin(allowed)].copy()
    del raw
    drop = [column for column in ("event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m") if column in data]
    feature = component.merge(data.drop(columns=drop), on="event_id", how="left", validate="one_to_one")
    kr = feature.market.eq("KR")
    feature["direction_score"] = np.where(
        kr,
        0.6 * feature.base + 0.4 * feature.category,
        0.4 * feature.base + 0.6 * feature.category,
    )
    feature["direction_cutoff"] = np.where(kr, 0.64, 0.60)
    feature["pred"] = (feature.direction_score >= feature.direction_cutoff).astype(int)
    feature["correct"] = feature.pred.eq(feature.y).astype(int)
    feature["signed_net"] = np.where(feature.pred.eq(1), 1.0, -1.0) * feature.fwd_ret_30m - ROUND_TRIP_COST
    feature["eval_weight"] = np.where(
        kr,
        KR_WEIGHT / kr.sum(),
        US_WEIGHT / (~kr).sum(),
    )
    feature["distance"] = (feature.direction_score - feature.direction_cutoff).abs()
    feature["down_distance"] = feature.direction_cutoff - feature.direction_score
    feature["up_distance"] = feature.direction_score - feature.direction_cutoff
    feature["abs_pre_ret_60"] = pd.to_numeric(feature.pre_ret_60, errors="coerce").abs()
    feature["abs_pre_ret_30"] = pd.to_numeric(feature.pre_ret_30, errors="coerce").abs()
    feature["abs_pre_ret_5"] = pd.to_numeric(feature.pre_ret_5, errors="coerce").abs()
    feature["abs_benchmark_30"] = pd.to_numeric(feature.benchmark_ret_30, errors="coerce").abs()

    print("LABEL-INDEPENDENT CONFIDENCE")
    for name in (
        "distance", "down_distance", "up_distance", "pre_vol_10", "pre_vol_30",
        "range_5m", "volume_ratio_5_30", "abs_pre_ret_60", "abs_pre_ret_30",
        "abs_pre_ret_5", "abs_benchmark_30",
    ):
        report(name, pd.to_numeric(feature[name], errors="coerce").fillna(-np.inf).to_numpy(float), feature)

    numeric = [
        "base", "category", "direction_score", "distance", "down_distance", "up_distance",
        "pre_ret_2", "pre_ret_5", "pre_ret_15", "pre_ret_30", "pre_ret_60",
        "pre_vol_10", "pre_vol_30", "volume_ratio_5_30", "range_5m",
        "minutes_from_open", "minutes_to_close", "entry_price",
        "event_positive_kw", "event_negative_kw", "event_financing_kw", "event_trial_kw",
        "event_regulatory_kw", "event_ma_kw", "event_earnings_kw", "headline_len", "body_len",
        "benchmark_ret_5", "benchmark_ret_30", "benchmark_ret_60",
    ]
    model_predictions: dict[str, np.ndarray] = {
        "correct_lgb": np.full(len(feature), np.nan),
        "positive_net_lgb": np.full(len(feature), np.nan),
        "signed_return_lgb": np.full(len(feature), np.nan),
    }
    for market in ("KR", "US"):
        positions = np.flatnonzero(feature.market.eq(market).to_numpy())
        part = feature.iloc[positions].copy().reset_index(drop=True)
        timestamp = pd.to_datetime(part.event_time_utc, utc=True)
        if market == "KR":
            group = part.event_id.astype(str).str.rsplit(":", n=1).str[-1]
        else:
            group = part.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")
        folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(part, part.correct, group)
        for train_index, valid_index in folds:
            train, valid = part.iloc[train_index], part.iloc[valid_index]
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            train_x = imputer.fit_transform(train[numeric])
            valid_x = imputer.transform(valid[numeric])
            settings = dict(
                n_estimators=90, num_leaves=7, learning_rate=0.025,
                min_child_samples=45 if market == "KR" else 18,
                subsample=0.8, colsample_bytree=0.75, reg_lambda=8.0,
                verbosity=-1, random_state=SEED, n_jobs=-1,
            )
            correct_model = LGBMClassifier(**settings).fit(train_x, train.correct)
            profitable = train.signed_net.gt(0).astype(int)
            profitable_model = LGBMClassifier(**settings).fit(train_x, profitable)
            return_model = LGBMRegressor(**settings, objective="huber").fit(train_x, train.signed_net)
            global_positions = positions[valid_index]
            model_predictions["correct_lgb"][global_positions] = correct_model.predict_proba(valid_x)[:, 1]
            model_predictions["positive_net_lgb"][global_positions] = profitable_model.predict_proba(valid_x)[:, 1]
            model_predictions["signed_return_lgb"][global_positions] = return_model.predict(valid_x)

    print("\nCROSS-FIT CONFIDENCE MODELS")
    for name, values in model_predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"missing model prediction: {name}")
        report(name, values, feature)
        if name == "correct_lgb":
            print("auc_correct", roc_auc_score(feature.correct, values, sample_weight=feature.eval_weight))
        if name == "positive_net_lgb":
            print("auc_positive_net", roc_auc_score(feature.signed_net.gt(0), values, sample_weight=feature.eval_weight))

    for correct_weight in (0.25, 0.5, 0.75):
        a = pd.Series(model_predictions["correct_lgb"]).rank(pct=True).to_numpy()
        b = pd.Series(model_predictions["signed_return_lgb"]).rank(pct=True).to_numpy()
        report(f"blend_{correct_weight:.2f}", correct_weight * a + (1.0 - correct_weight) * b, feature)

    output = feature[[
        "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
        "base", "category", "direction_score", "direction_cutoff", "pred", "correct",
        "signed_net", "eval_weight", "distance",
    ]].copy()
    for name, values in model_predictions.items():
        output[name] = values
    output.to_parquet(ROOT / "cache" / "v13_confidence_oof.parquet", index=False)


if __name__ == "__main__":
    main()
