"""Three-way opened V15 DEV/V15 SEAL/V16 rank-feature stability search."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app


def rank(series: pd.Series) -> np.ndarray:
    value=pd.to_numeric(series,errors="coerce")
    value=value.fillna(value.median() if value.notna().any() else 0.0)
    return value.rank(method="average",pct=True).to_numpy(float)


def main() -> None:
    raw=pd.read_csv(
        app.DATA/"labeled_events_v16.csv.gz",compression="gzip",dtype={"ticker":str},
        parse_dates=["event_time_utc"],low_memory=False,
    )
    frame=app.model_frame(raw);frame["y"]=raw.y.to_numpy();frame["source"]=raw.source.to_numpy()
    source_groups={
        "US":("SEC_V15_DEV","SEC_V15_SEAL","SEC_V16_SEAL"),
        "KR":("KIND_V15_DEV","KIND_V15_SEAL","KIND_V16_SEAL"),
    }
    for market,sources in source_groups.items():
        parts={source:frame[frame.source.eq(source)].reset_index(drop=True) for source in sources}
        feature_scores={};single=[]
        for column in app.NUMERIC_COLS:
            raw_scores={source:rank(part[column]) for source,part in parts.items()}
            raw_aucs=[roc_auc_score(parts[source].y,raw_scores[source]) for source in sources]
            sign=1.0 if np.mean(raw_aucs)>=.5 else -1.0
            scores={source:sign*raw_scores[source] for source in sources}
            aucs=[float(roc_auc_score(parts[source].y,scores[source])) for source in sources]
            feature_scores[column]=(sign,scores)
            single.append({"features":column,"signs":str(sign),
                           **{f"auc_{i+1}":a for i,a in enumerate(aucs)},
                           "min_auc":min(aucs),"mean_auc":float(np.mean(aucs))})
        singles=pd.DataFrame(single).sort_values(["min_auc","mean_auc"],ascending=False)
        top=singles.sort_values("mean_auc",ascending=False).head(24).features.tolist()
        rows=[]
        for size in (1,2,3,4):
            for combo in itertools.combinations(top,size):
                aucs=[]
                for source in sources:
                    score=np.mean([feature_scores[column][1][source] for column in combo],axis=0)
                    aucs.append(float(roc_auc_score(parts[source].y,score)))
                rows.append({
                    "features":"|".join(combo),
                    "signs":"|".join(str(feature_scores[c][0]) for c in combo),
                    **{f"auc_{i+1}":a for i,a in enumerate(aucs)},
                    "min_auc":min(aucs),"mean_auc":float(np.mean(aucs)),
                })
        result=pd.DataFrame(rows).sort_values(["min_auc","mean_auc"],ascending=False)
        singles.to_csv(app.CACHE/f"v17_threeway_single_{market}.csv",index=False)
        result.to_csv(app.CACHE/f"v17_threeway_blend_{market}.csv",index=False)
        print(f"\n{market} SINGLES")
        print(singles.head(25).to_string(index=False))
        print(f"\n{market} BLENDS")
        print(result.head(30).to_string(index=False))


if __name__=="__main__":
    main()
