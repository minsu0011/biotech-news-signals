"""Train only on opened prior versions and hold V18 DEV out entirely."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


SPECS=[
    *[{"kind":"logit","C":c,"class_weight":w} for c in (.003,.01,.03,.08,.15,.35) for w in (None,"balanced")],
    *[{"kind":"native_cat","iterations":n,"depth":d,"cat_cols":["event_type","ticker"]}
      for n,d in ((150,2),(250,2),(250,3),(400,3),(300,4))],
    *[{"kind":"lgb","n_estimators":n,"num_leaves":leaves,"cat_cols":["event_type","ticker"]}
      for n,leaves in ((80,3),(120,5),(180,7),(250,11))],
]


def audits(target: pd.DataFrame,probability: np.ndarray) -> dict:
    out=target.copy();out["raw_prob"]=probability
    out["half"]=out.event_id.astype(str).map(
        lambda v:int(hashlib.sha256(f"V18-AUDIT|{v}".encode()).hexdigest(),16)%2
    )
    best=None
    for cutoff in np.arange(.30,.701,.01):
        p=np.clip(.5+.45*(probability-cutoff)/max(cutoff,1-cutoff),.02,.98)
        out["prob"]=p;mets={}
        for name,x in (("all",out),("half0",out[out.half.eq(0)]),("half1",out[out.half.eq(1)])):
            mets[name]=app.evaluate(x,.925,.002)
        min_bal=min(x["balanced_accuracy"] for x in mets.values())
        min_edge=min(x["edge_vs_naive"] for x in mets.values())
        min_auc=min(x["auc"] for x in mets.values())
        score=3*min_bal+2*min_auc+min_edge+mets["all"]["balanced_accuracy"]
        row={"cutoff":float(cutoff),"score":score,"min_bal":min_bal,
             "min_edge":min_edge,"min_auc":min_auc,"metrics":mets}
        if best is None or row["score"]>best["score"]:best=row
    return best


def main():
    cfg=app.Config();data=app.load_search_frame(cfg);dev,_,_=app.split_dev_seal(data,cfg)
    for market,source in (("US","SEC_V18_DEV"),("KR","KIND_V18_DEV")):
        target=dev[dev.market.eq(market)&dev.source.eq(source)&dev.y.notna()].copy().reset_index(drop=True)
        opened=dev[dev.market.eq(market)&~dev.source.eq(source)&dev.y.notna()].copy()
        rows=[]
        for scope in ("all","same_tickers"):
            prior=opened if scope=="all" else opened[opened.ticker.astype(str).isin(set(target.ticker.astype(str)))]
            for spec in SPECS:
                try:
                    model=app.make_model(spec);model.fit(app.model_frame(prior),prior.y.astype(int))
                    probability=model.predict_proba(app.model_frame(target))[:,1]
                    audit=audits(target,probability)
                    rows.append({"scope":scope,"prior_n":len(prior),"spec":spec,**audit})
                except Exception as exc:
                    rows.append({"scope":scope,"spec":spec,"error":str(exc),"score":-1})
        rows.sort(key=lambda x:x["score"],reverse=True)
        print(json.dumps({"market":market,"top":rows[:20]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
