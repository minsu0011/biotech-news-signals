"""Joint market cutoff search on repeated V15 DEV OOF directions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def main()->None:
    raw=pd.read_parquet(app.CACHE/"v15_repeated_oof.parquet")
    candidates={
        "US_native":raw[raw.market.eq("US")&~raw.augment&raw.model.eq("native")].copy(),
        "US_cat3":raw[raw.market.eq("US")&~raw.augment&raw.model.eq("cat3")].copy(),
        "KR_aug_logit001":raw[raw.market.eq("KR")&raw.augment&raw.model.eq("logit001")].copy(),
        "KR_logit001":raw[raw.market.eq("KR")&~raw.augment&raw.model.eq("logit001")].copy(),
        "KR_logit003":raw[raw.market.eq("KR")&~raw.augment&raw.model.eq("logit003")].copy(),
    }
    rows=[]
    for us_name in ("US_native","US_cat3"):
        for kr_name in ("KR_aug_logit001","KR_logit001","KR_logit003"):
            us=candidates[us_name];kr=candidates[kr_name]
            for us_cutoff in np.arange(.45,.651,.01):
                for kr_cutoff in np.arange(.45,.651,.01):
                    left=us.copy();right=kr.copy()
                    left["prob"]=app.shift_probability(left.prob.to_numpy(float),float(us_cutoff))
                    right["prob"]=app.shift_probability(right.prob.to_numpy(float),float(kr_cutoff))
                    left["eval_weight"]=448.0/len(left);right["eval_weight"]=1521.0/len(right)
                    metrics=app.evaluate(pd.concat([left,right],ignore_index=True),.75,.002)
                    by=metrics["by_market"]
                    passes=sum([
                        metrics["balanced_accuracy"]>=.54,metrics["auc"]>=.56,
                        metrics["edge_vs_naive"]>=.02,
                        by["US"]["balanced_accuracy"]>=.51,
                        by["KR"]["balanced_accuracy"]>=.51,
                    ])
                    score=(metrics["balanced_accuracy"]+metrics["auc"]+metrics["accuracy"]+
                           min(metrics["edge_vs_naive"],.06))
                    rows.append((passes,score,us_name,kr_name,float(us_cutoff),float(kr_cutoff),metrics))
    rows.sort(key=lambda item:(item[0],item[1]),reverse=True)
    for item in rows[:20]:
        passes,score,us_name,kr_name,us_cutoff,kr_cutoff,metrics=item
        print(json.dumps({
            "passes":passes,"score":score,"US":us_name,"KR":kr_name,
            "US_cutoff":us_cutoff,"KR_cutoff":kr_cutoff,
            "accuracy":metrics["accuracy"],"balanced_accuracy":metrics["balanced_accuracy"],
            "auc":metrics["auc"],"edge":metrics["edge_vs_naive"],
            "by_market":metrics["by_market"],
        }))


if __name__=="__main__":
    main()
