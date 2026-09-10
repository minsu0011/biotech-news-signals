#!/usr/bin/env python
"""DEV-only diagnostics for V13.  Never evaluates a V13 SEAL row."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "labeled_events_v13.csv.gz"
SEED = 260825


def normalize_headline(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\d+(?:[.,]\d+)*", " N ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rank_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    val = np.asarray(values, dtype=float)
    out = np.full(len(val), 0.5)
    valid = np.isfinite(val)
    out[valid] = np.searchsorted(ref, val[valid], side="right") / len(ref)
    return out


def category_prediction(train: pd.DataFrame, valid: pd.DataFrame, column: str, smoothing: float) -> np.ndarray:
    global_rate = float(train.y.mean())
    stats = train.groupby(column).y.agg(["sum", "count"])
    mapping = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
    return valid[column].map(mapping).fillna(global_rate).to_numpy(float)


def smooth_mapping(train: pd.DataFrame, column: str, smoothing: float) -> tuple[pd.Series, float]:
    global_rate = float(train.y.mean())
    stats = train.groupby(column).y.agg(["sum", "count"])
    mapping = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
    return mapping, global_rate


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def main() -> None:
    use = [
        "event_id", "event_time_utc", "market", "source", "ticker", "headline", "text",
        "y", "fwd_ret_30m", "pre_ret_5", "pre_ret_30", "pre_ret_60",
        "pre_vol_10", "volume_ratio_5_30", "minutes_from_open",
        "benchmark_ret_5", "benchmark_ret_30",
    ]
    raw = pd.read_csv(DATA, usecols=use)
    # This is the hard guard: no V13 SEAL source can enter any diagnostic below.
    allowed = {"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"}
    if raw.source.isin({"NAVER_NEWS_V13_SEAL", "SEC_V13_SEAL"}).all():
        raise RuntimeError("unexpected input: only sealed rows")
    data = raw[raw.source.isin(allowed)].copy()
    del raw
    if set(data.source) != allowed:
        raise RuntimeError(f"DEV source missing: {set(data.source)}")

    kr = data[data.source.eq("NAVER_NEWS_V13_DEV")].copy().reset_index(drop=True)
    kr["article"] = kr.event_id.astype(str).str.rsplit(":", n=1).str[-1]
    kr["headnorm"] = kr.headline.map(normalize_headline)
    groups = kr.article
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    folds = list(splitter.split(kr, kr.y, groups))

    print(f"KR n={len(kr)} up={kr.y.mean():.6f} articles={kr.article.nunique()} headnorm={kr.headnorm.nunique()}")
    print("headnorm_repeat_rows", int(kr.headnorm.duplicated(keep=False).sum()))
    for smoothing in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        p = np.full(len(kr), np.nan)
        seen = np.zeros(len(kr), dtype=bool)
        fold_auc = []
        for tr_idx, va_idx in folds:
            tr, va = kr.iloc[tr_idx], kr.iloc[va_idx]
            pred = category_prediction(tr, va, "headnorm", smoothing)
            p[va_idx] = pred
            seen[va_idx] = va.headnorm.isin(set(tr.headnorm))
            fold_auc.append(roc_auc_score(va.y, 1.0 - pred))
        print(
            "HEAD",
            f"k={smoothing:g}",
            f"inverse_auc={roc_auc_score(kr.y, 1.0-p):.6f}",
            f"folds={[round(x, 4) for x in fold_auc]}",
            f"seen={seen.mean():.4f}",
        )

    cached = pd.read_parquet(ROOT / "cache" / "v13_target_oof.parquet")
    target = cached[cached.market.eq("KR")].copy()
    target = target.merge(kr[["event_id", "headnorm"]], on="event_id", how="left", validate="one_to_one")
    target_index = {event_id: i for i, event_id in enumerate(target.event_id)}
    head_inv = np.full(len(target), np.nan)
    for tr_idx, va_idx in folds:
        tr, va = kr.iloc[tr_idx], kr.iloc[va_idx]
        pred = 1.0 - category_prediction(tr, va, "headnorm", 5.0)
        positions = [target_index[event_id] for event_id in va.event_id]
        head_inv[positions] = pred
    target["head_inv"] = head_inv
    if target[["raw", "head_inv"]].isna().any().any():
        raise RuntimeError("OOF merge produced missing values")

    # Rank blending is fit independently in each fold to match production ECDF behavior.
    for weight in (0.15, 0.25, 0.35, 0.5, 0.65, 0.8):
        score = np.full(len(target), np.nan)
        for tr_idx, va_idx in folds:
            train_ids = set(kr.iloc[tr_idx].event_id)
            valid_ids = set(kr.iloc[va_idx].event_id)
            ti = target.event_id.isin(train_ids)
            vi = target.event_id.isin(valid_ids)
            base_rank = rank_against(target.loc[ti, "raw"].to_numpy(), target.loc[vi, "raw"].to_numpy())
            head_rank = rank_against(target.loc[ti, "head_inv"].to_numpy(), target.loc[vi, "head_inv"].to_numpy())
            score[vi] = (1.0 - weight) * base_rank + weight * head_rank
        print("BLEND", f"head_weight={weight:.2f}", f"auc={roc_auc_score(target.y, score):.6f}")

    # Fold- and half-sample diagnostics prevent selecting a one-bucket accident.
    best_score = np.full(len(target), np.nan)
    for tr_idx, va_idx in folds:
        train_ids = set(kr.iloc[tr_idx].event_id)
        valid_ids = set(kr.iloc[va_idx].event_id)
        ti = target.event_id.isin(train_ids)
        vi = target.event_id.isin(valid_ids)
        a = rank_against(target.loc[ti, "raw"].to_numpy(), target.loc[vi, "raw"].to_numpy())
        b = rank_against(target.loc[ti, "head_inv"].to_numpy(), target.loc[vi, "head_inv"].to_numpy())
        best_score[vi] = 0.65 * a + 0.35 * b
    target["score"] = best_score
    target["fold_check"] = 0
    for fold_number, (_, va_idx) in enumerate(folds, 1):
        valid_ids = set(kr.iloc[va_idx].event_id)
        target.loc[target.event_id.isin(valid_ids), "fold_check"] = fold_number
    target["year_half"] = pd.to_datetime(target.event_time_utc, utc=True).dt.strftime("%Y-H") + np.where(
        pd.to_datetime(target.event_time_utc, utc=True).dt.month <= 6, "1", "2"
    )
    for fold, part in target.groupby("fold_check"):
        print("STABILITY", "fold", fold, "n", len(part), "auc", round(roc_auc_score(part.y, part.score), 6))
    for period, part in target.groupby("year_half"):
        if part.y.nunique() > 1 and len(part) >= 25:
            print("STABILITY", period, "n", len(part), "auc", round(roc_auc_score(part.y, part.score), 6))

    print("\nCATEGORY TARGET ENCODING")
    for market, part in data.groupby("market"):
        part = part.copy().reset_index(drop=True)
        part["article"] = part.event_id.astype(str).str.rsplit(":", n=1).str[-1]
        part["headnorm"] = part.headline.map(normalize_headline)
        timestamp = pd.to_datetime(part.event_time_utc, utc=True)
        part["hour_bin"] = (timestamp.dt.hour * 2 + timestamp.dt.minute // 30).astype(str)
        part["ticker_hour"] = part.ticker.astype(str) + "|" + part.hour_bin
        if market == "KR":
            group = part.article
        else:
            group = part.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")
        cv = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(part, part.y, group))
        for column in ("ticker", "hour_bin", "ticker_hour", "headnorm"):
            for smoothing in (2.0, 10.0, 50.0):
                prediction = np.full(len(part), np.nan)
                seen = np.zeros(len(part), dtype=bool)
                within = []
                for train_index, valid_index in cv:
                    train, valid = part.iloc[train_index], part.iloc[valid_index]
                    fold_prediction = category_prediction(train, valid, column, smoothing)
                    prediction[valid_index] = fold_prediction
                    seen[valid_index] = valid[column].isin(set(train[column]))
                    within.append(roc_auc_score(valid.y, fold_prediction))
                direct = roc_auc_score(part.y, prediction)
                stable = float(np.mean(within))
                orientation = "direct" if stable >= 0.5 else "inverse"
                oriented = direct if orientation == "direct" else 1.0 - direct
                print(
                    market, column, f"k={smoothing:g}", orientation,
                    f"combined={oriented:.6f}", f"within_mean={max(stable,1-stable):.6f}",
                    f"folds={[round(x if orientation == 'direct' else 1-x, 4) for x in within]}",
                    f"seen={seen.mean():.3f}",
                )

    print("\nHONEST V13 COMPONENT BLENDS")
    component_rows = []
    for market, part in data.groupby("market"):
        part = part.copy().reset_index(drop=True)
        timestamp = pd.to_datetime(part.event_time_utc, utc=True)
        part["article"] = part.event_id.astype(str).str.rsplit(":", n=1).str[-1]
        part["hour_bin"] = (timestamp.dt.hour * 2 + timestamp.dt.minute // 30).astype(str)
        part["ticker_hour"] = part.ticker.astype(str) + "|" + part.hour_bin
        if market == "KR":
            group = part.article
        else:
            group = part.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")
        cv = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(part, part.y, group))
        base = np.full(len(part), np.nan)
        category = np.full(len(part), np.nan)
        confidence_q = np.full(len(part), np.nan)
        for train_index, valid_index in cv:
            train, valid = part.iloc[train_index], part.iloc[valid_index]
            if market == "KR":
                vectorizer = TfidfVectorizer(
                    lowercase=True, ngram_range=(1, 2), min_df=2,
                    max_features=50000, sublinear_tf=True,
                )
                train_text = vectorizer.fit_transform(train.text.fillna(""))
                model = LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000,
                    solver="liblinear", random_state=SEED,
                ).fit(train_text, train.y)
                train_signals = [1.0 - model.predict_proba(train_text)[:, 1]]
                valid_signals = [1.0 - model.predict_proba(vectorizer.transform(valid.text.fillna("")))[:, 1]]
                for column, sign in (
                    ("pre_ret_5", -1.0), ("pre_ret_30", -1.0),
                    ("pre_vol_10", 1.0), ("volume_ratio_5_30", 1.0),
                ):
                    train_signals.append(sign * pd.to_numeric(train[column], errors="coerce").to_numpy(float))
                    valid_signals.append(sign * pd.to_numeric(valid[column], errors="coerce").to_numpy(float))
                train_base = np.mean([
                    rank_against(train_score, train_score)
                    for train_score in train_signals
                ], axis=0)
                valid_base = np.mean([
                    rank_against(train_score, valid_score)
                    for train_score, valid_score in zip(train_signals, valid_signals)
                ], axis=0)
                base[valid_index] = valid_base
                mapping, prior = smooth_mapping(train, "hour_bin", 20.0)
                train_category = 1.0 - train.hour_bin.map(mapping).fillna(prior).to_numpy(float)
                valid_category = 1.0 - valid.hour_bin.map(mapping).fillna(prior).to_numpy(float)
            else:
                columns = ["benchmark_ret_30", "pre_ret_60", "minutes_from_open"]
                signs = np.asarray([1.0, -1.0, -1.0])
                train_values = train[columns].apply(pd.to_numeric, errors="coerce")
                medians = train_values.median().fillna(0.0)
                train_values = train_values.fillna(medians)
                valid_values = valid[columns].apply(pd.to_numeric, errors="coerce").fillna(medians)
                means = train_values.mean()
                standard_deviations = train_values.std().replace(0, np.nan).fillna(1.0)
                train_context = (((train_values - means) / standard_deviations) * signs).mean(axis=1).to_numpy(float)
                valid_context = (((valid_values - means) / standard_deviations) * signs).mean(axis=1).to_numpy(float)
                train_base = rank_against(train_context, train_context)
                valid_base = rank_against(train_context, valid_context)
                base[valid_index] = valid_base
                mapping, prior = smooth_mapping(train, "ticker_hour", 2.0)
                train_category = 1.0 - train.ticker_hour.map(mapping).fillna(prior).to_numpy(float)
                valid_category = 1.0 - valid.ticker_hour.map(mapping).fillna(prior).to_numpy(float)
            train_category_rank = rank_against(train_category, train_category)
            valid_category_rank = rank_against(train_category, valid_category)
            category[valid_index] = valid_category_rank
            category_weight = 0.4 if market == "KR" else 0.6
            direction_cutoff = 0.64 if market == "KR" else 0.60
            train_direction = (1.0 - category_weight) * train_base + category_weight * train_category_rank
            valid_direction = (1.0 - category_weight) * valid_base + category_weight * valid_category_rank
            train_distance = rank_against(
                np.abs(train_direction - direction_cutoff),
                np.abs(train_direction - direction_cutoff),
            )
            valid_distance = rank_against(
                np.abs(train_direction - direction_cutoff),
                np.abs(valid_direction - direction_cutoff),
            )
            train_benchmark = pd.to_numeric(train.benchmark_ret_5, errors="coerce").abs().to_numpy(float)
            valid_benchmark = pd.to_numeric(valid.benchmark_ret_5, errors="coerce").abs().to_numpy(float)
            train_benchmark_rank = rank_against(train_benchmark, train_benchmark)
            valid_benchmark_rank = rank_against(train_benchmark, valid_benchmark)
            train_confidence = train_distance + 1.4 * train_benchmark_rank - 0.4 * (train_direction >= direction_cutoff)
            valid_confidence = valid_distance + 1.4 * valid_benchmark_rank - 0.4 * (valid_direction >= direction_cutoff)
            confidence_q[valid_index] = rank_against(train_confidence, valid_confidence)
        if np.isnan(base).any() or np.isnan(category).any() or np.isnan(confidence_q).any():
            raise RuntimeError(f"missing component OOF: {market}")
        for category_weight in np.arange(0.0, 0.81, 0.1):
            combined = (1.0 - category_weight) * base + category_weight * category
            print(
                "COMPONENT", market, f"cat_weight={category_weight:.1f}",
                f"auc={roc_auc_score(part.y, combined):.6f}",
            )
        z = part[["event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m"]].copy()
        z["base"] = base
        z["category"] = category
        z["confidence_q"] = confidence_q
        component_rows.append(z)

    components = pd.concat(component_rows, ignore_index=True)
    # Coarse but fully DEV-only joint search. Raw ranks are converted to a
    # monotone probability only after choosing one cutoff per market.
    best = []
    for kr_weight in (0.2, 0.3, 0.4, 0.5):
        for us_weight in (0.3, 0.4, 0.5, 0.6):
            score = components.base.copy()
            score.loc[components.market.eq("KR")] = (
                (1.0 - kr_weight) * components.loc[components.market.eq("KR"), "base"]
                + kr_weight * components.loc[components.market.eq("KR"), "category"]
            )
            score.loc[components.market.eq("US")] = (
                (1.0 - us_weight) * components.loc[components.market.eq("US"), "base"]
                + us_weight * components.loc[components.market.eq("US"), "category"]
            )
            for kr_cut in np.arange(0.54, 0.76, 0.02):
                for us_cut in np.arange(0.48, 0.76, 0.02):
                    cutoff = np.where(components.market.eq("KR"), kr_cut, us_cut)
                    probability = sigmoid(10.0 * (score.to_numpy(float) - cutoff))
                    prediction = probability >= 0.5
                    weight = np.where(components.market.eq("KR"), 761.0 / (components.market == "KR").sum(), 350.0 / (components.market == "US").sum())
                    accuracy = accuracy_score(components.y, prediction, sample_weight=weight)
                    up_rate = np.average(components.y, weights=weight)
                    edge = accuracy - max(up_rate, 1.0 - up_rate)
                    balance = balanced_accuracy_score(components.y, prediction, sample_weight=weight)
                    market_balance = {
                        market: balanced_accuracy_score(
                            components.loc[components.market.eq(market), "y"],
                            prediction[components.market.eq(market)],
                        )
                        for market in ("KR", "US")
                    }
                    auc = roc_auc_score(components.y, probability, sample_weight=weight)
                    passed = sum((balance >= 0.54, auc >= 0.56, edge >= 0.02, market_balance["KR"] >= 0.51, market_balance["US"] >= 0.51))
                    best.append((passed, balance + auc + edge, kr_weight, us_weight, kr_cut, us_cut, accuracy, balance, auc, edge, market_balance))
    best.sort(reverse=True, key=lambda value: (value[0], value[1]))
    print("\nBEST DIRECTION SETTINGS")
    for row in best[:15]:
        print(row)
    fixed_score = np.where(
        components.market.eq("KR"),
        0.6 * components.base + 0.4 * components.category,
        0.4 * components.base + 0.6 * components.category,
    )
    fixed_cutoff = np.where(components.market.eq("KR"), 0.64, 0.60)
    fixed_pred = fixed_score >= fixed_cutoff
    strength = np.where(components.market.eq("KR"), components.confidence_q, 0.5 * components.confidence_q)
    fixed_probability = np.where(fixed_pred, 0.51 + 0.49 * strength, 0.49 - 0.49 * strength)
    fixed_weight = np.where(components.market.eq("KR"), 761.0 / (components.market == "KR").sum(), 350.0 / (components.market == "US").sum())
    fixed_high = np.maximum(fixed_probability, 1.0 - fixed_probability) >= 0.89
    fixed_correct = fixed_pred == components.y.to_numpy(int)
    fixed_net = np.where(fixed_pred, 1.0, -1.0) * components.fwd_ret_30m.to_numpy(float) - 0.002
    print(
        "PRODUCTION_SAFE_FIXED",
        "accuracy", accuracy_score(components.y, fixed_pred, sample_weight=fixed_weight),
        "balance", balanced_accuracy_score(components.y, fixed_pred, sample_weight=fixed_weight),
        "auc", roc_auc_score(components.y, fixed_probability, sample_weight=fixed_weight),
        "coverage", fixed_weight[fixed_high].sum() / fixed_weight.sum(),
        "hc_accuracy", np.average(fixed_correct[fixed_high], weights=fixed_weight[fixed_high]),
        "hc_net", np.average(fixed_net[fixed_high], weights=fixed_weight[fixed_high]),
    )
    components.to_parquet(ROOT / "cache" / "v13_components_oof.parquet", index=False)


if __name__ == "__main__":
    main()
