"""Label-safe V18 DEV feature and conventional-model scan."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bio_news_30m_v3 as app


def oriented_auc(y: pd.Series, value: pd.Series) -> tuple[float, int]:
    x=pd.to_numeric(value,errors="coerce")
    fill=float(x.median()) if x.notna().any() else 0.0
    x=x.fillna(fill).to_numpy(float);yy=y.astype(int).to_numpy()
    auc=float(roc_auc_score(yy,x))
    return (auc,1) if auc>=.5 else (1.0-auc,-1)


def main() -> None:
    cfg=app.Config();all_data=app.load_search_frame(cfg);dev,_,_=app.split_dev_seal(all_data,cfg)
    for market,source in (("US","SEC_V18_DEV"),("KR","KIND_V18_DEV")):
        raw=dev[dev.market.eq(market)&dev.source.eq(source)&dev.y.notna()].copy().reset_index(drop=True)
        frame=app.model_frame(raw)
        half=raw.event_id.astype(str).map(
            lambda value:int(hashlib.sha256(f"V18-AUDIT|{value}".encode()).hexdigest(),16)%2
        )
        rows=[]
        for column in app.NUMERIC_COLS:
            all_auc,sign=oriented_auc(raw.y,frame[column])
            half_auc=[]
            for value in (0,1):
                auc,_=oriented_auc(raw.loc[half.eq(value),"y"],frame.loc[half.eq(value),column])
                # Keep the orientation selected on all DEV fixed in the audit halves.
                direct=float(roc_auc_score(
                    raw.loc[half.eq(value),"y"].astype(int),
                    sign*pd.to_numeric(frame.loc[half.eq(value),column],errors="coerce").fillna(
                        pd.to_numeric(frame[column],errors="coerce").median()
                    ),
                ))
                half_auc.append(direct)
            rows.append({"column":column,"auc":all_auc,"sign":sign,"half0":half_auc[0],"half1":half_auc[1],"min_half":min(half_auc)})
        rows.sort(key=lambda item:(item["min_half"],item["auc"]),reverse=True)
        print(json.dumps({"market":market,"n":len(raw),"up_rate":float(raw.y.mean()),"features":rows[:20]},indent=2))

        candidates=[]
        for c in (.01,.03,.08,.15,.35,.7,1.5):
            for weight in (None,"balanced"):
                candidates.append({"kind":"logit","C":c,"class_weight":weight})
        candidates += [
            {"kind":"native_cat","iterations":250,"depth":d,"cat_cols":["event_type","ticker"]}
            for d in (2,3,4)
        ]
        candidates += [
            {"kind":"lgb","n_estimators":n,"num_leaves":leaves,"cat_cols":["event_type","ticker"]}
            for n,leaves in ((100,5),(180,7),(250,11))
        ]
        model_rows=[]
        for spec in candidates:
            try:
                pred=app.cv_market_predictions(dev,market,spec,cfg)
                model_rows.append({"spec":spec,"metrics":app.evaluate(pred,.70,cfg.round_trip_cost)})
            except Exception as exc:
                model_rows.append({"spec":spec,"error":str(exc)})
        model_rows.sort(key=lambda item:item.get("metrics",{}).get("auc",-1),reverse=True)
        print(json.dumps({"market":market,"models":model_rows},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
