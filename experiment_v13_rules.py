#!/usr/bin/env python
"""DEV-only robust rank-rule search using only entry-time features."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import bio_news_30m_v3 as app


RULES = {
    "KR": {
        "close10_rev": ("close_position_10", -1.0),
        "ret20_rev": ("pre_ret_20", -1.0),
        "ret45_rev": ("pre_ret_45", -1.0),
        "ret30_rev": ("pre_ret_30", -1.0),
        "ret5_rev": ("pre_ret_5", -1.0),
        "entry_rev": ("entry_bar_ret", -1.0),
        "bench60_rev": ("benchmark_ret_60", -1.0),
        "ret120_rev": ("pre_ret_120", -1.0),
        "ret15_rev": ("pre_ret_15", -1.0),
        "volume10": ("volume_ratio_10_60", 1.0),
    },
    "US": {
        "ret120_rev": ("pre_ret_120", -1.0),
        "ret90_rev": ("pre_ret_90", -1.0),
        "bench30": ("benchmark_ret_30", 1.0),
        "excess60_rev": ("excess_ret_60", -1.0),
        "ret60_rev": ("pre_ret_60", -1.0),
        "close10_rev": ("close_position_10", -1.0),
        "trend30": ("trend_slope_30", 1.0),
        "volume5": ("volume_ratio_5_30", 1.0),
        "ma_kw": ("event_ma_kw", 1.0),
        "positive_kw": ("event_positive_kw", 1.0),
    },
}


def main() -> None:
    raw = pd.read_csv(app.LABELED_FILE, dtype={"ticker": str}, low_memory=False)
    allowed = {"NAVER_NEWS_V13_DEV", "SEC_V13_DEV"}
    raw = raw[raw.source.isin(allowed)].copy().reset_index(drop=True)
    features = app.model_frame(raw)
    for market in ("KR", "US"):
        part = raw[raw.market.eq(market)].copy().reset_index(drop=True)
        x = features[raw.market.eq(market)].reset_index(drop=True)
        timestamp = pd.to_datetime(part.event_time_utc, utc=True)
        if market == "KR":
            groups = part.event_id.astype(str).str.rsplit(":", n=1).str[-1]
            period = timestamp.dt.strftime("%Y-") + np.where(timestamp.dt.month <= 6, "H1", "H2")
        else:
            groups = part.ticker.astype(str) + "|" + timestamp.dt.strftime("%Y-%m-%d")
            period = timestamp.dt.year.astype(str)
        folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=260825).split(part, part.y, groups))
        signals = {}
        for name, (column, sign) in RULES[market].items():
            values = sign * pd.to_numeric(x[column], errors="coerce")
            signals[name] = values.rank(pct=True).fillna(0.5).to_numpy(float)
        candidates = []
        names = list(signals)
        for size in range(1, 6):
            for chosen in itertools.combinations(names, size):
                score = np.mean([signals[name] for name in chosen], axis=0)
                overall = roc_auc_score(part.y, score)
                fold_auc = [roc_auc_score(part.iloc[valid].y, score[valid]) for _, valid in folds]
                period_auc = [
                    roc_auc_score(part.loc[period.eq(bucket), "y"], score[period.eq(bucket)])
                    for bucket in sorted(period.unique())
                    if part.loc[period.eq(bucket), "y"].nunique() > 1
                ]
                robust = min(fold_auc + period_auc)
                candidates.append((robust, np.mean(fold_auc + period_auc), overall, chosen, fold_auc, period_auc))
        candidates.sort(reverse=True, key=lambda row: (row[0], row[1], row[2]))
        print("\n", market, "ROBUST")
        for row in candidates[:25]:
            print(row)
        print("\n", market, "OVERALL")
        for row in sorted(candidates, reverse=True, key=lambda value: value[2])[:25]:
            print(row)


if __name__ == "__main__":
    main()
