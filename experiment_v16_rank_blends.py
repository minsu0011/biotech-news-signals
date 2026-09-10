"""Stable opened-V15 rank-blend search for V16, evaluated on both old splits."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app


FEATURES = {
    "US":[
        "intraday_ret_open","benchmark_ret_5","pre_ret_90","range_30m",
        "range_60m","excess_ret_30","pre_ret_120","range_10m",
        "pre_vol_10","vwap_distance_30","pre_vol_60","pre_ret_60",
        "ticker_te",
    ],
    "KR":[
        "close_position_10","pre_ret_30","entry_bar_ret","pre_ret_60",
        "excess_ret_5","pre_ret_20","excess_ret_60","pre_ret_15",
        "excess_ret_15","pre_ret_10","excess_ret_30","pre_ret_5",
    ],
}


def rank(values: pd.Series) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce")
    x = x.fillna(x.median() if x.notna().any() else 0.0)
    return x.rank(method="average", pct=True).to_numpy(float)


def target_encode(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    base=float(train.y.mean())
    stat=train.groupby("ticker").y.agg(["mean","count"])
    score=(stat["mean"]*stat["count"]+base*2.0)/(stat["count"]+2.0)
    return test.ticker.map(score).fillna(base).to_numpy(float)


def main() -> None:
    raw=pd.read_csv(
        app.DATA/"labeled_events_v15.csv.gz",compression="gzip",dtype={"ticker":str},
        parse_dates=["event_time_utc"],low_memory=False,
    )
    data=app.model_frame(raw)
    data["y"]=raw.y.to_numpy();data["source"]=raw.source.to_numpy()
    for market,dev_source,seal_source in (
        ("US","SEC_V15_DEV","SEC_V15_SEAL"),
        ("KR","KIND_V15_DEV","KIND_V15_SEAL"),
    ):
        dev=data[data.source.eq(dev_source)].copy().reset_index(drop=True)
        seal=data[data.source.eq(seal_source)].copy().reset_index(drop=True)
        score_dev={};score_seal={};signs={}
        for feature in FEATURES[market]:
            if feature=="ticker_te":
                left=rank(pd.Series(target_encode(seal,dev)))
                right=rank(pd.Series(target_encode(dev,seal)))
            else:
                left=rank(dev[feature]);right=rank(seal[feature])
            sign=1.0 if roc_auc_score(dev.y,left)>=.5 else -1.0
            score_dev[feature]=sign*left;score_seal[feature]=sign*right;signs[feature]=sign
        rows=[]
        for size in (1,2,3,4):
            for combo in itertools.combinations(FEATURES[market],size):
                weight_sets=[np.ones(size)]
                if size==2:
                    weight_sets.extend([np.array([.25,.75]),np.array([.75,.25])])
                for weights in weight_sets:
                    left=sum(w*score_dev[f] for w,f in zip(weights,combo))/weights.sum()
                    right=sum(w*score_seal[f] for w,f in zip(weights,combo))/weights.sum()
                    dev_auc=float(roc_auc_score(dev.y,left));seal_auc=float(roc_auc_score(seal.y,right))
                    rows.append({
                        "features":"|".join(combo),"weights":"|".join(map(str,weights)),
                        "signs":"|".join(str(signs[f]) for f in combo),
                        "dev_auc":dev_auc,"seal_auc":seal_auc,
                        "min_auc":min(dev_auc,seal_auc),"mean_auc":.5*(dev_auc+seal_auc),
                    })
        result=pd.DataFrame(rows).sort_values(["min_auc","mean_auc"],ascending=False)
        result.to_csv(app.CACHE/f"v16_rank_blends_{market}.csv",index=False)
        print(f"\n{market} TOP")
        print(result.head(30).to_string(index=False))


if __name__=="__main__":
    main()
