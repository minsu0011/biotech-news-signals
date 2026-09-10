"""Final confidence search around prior-only V18 transfer models."""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


MODEL={
    "US":{"spec":{"kind":"native_cat","iterations":150,"depth":2,"cat_cols":["event_type","ticker"]},"cutoff":.46},
    "KR":{"spec":{"kind":"lgb","n_estimators":180,"num_leaves":7,"cat_cols":["event_type","ticker"]},"cutoff":.53},
}
TARGET={"US":395.0,"KR":633.0}
OPPORTUNITY=[
    "pre_ret_2","pre_ret_5","pre_ret_10","pre_ret_15","pre_ret_30","pre_ret_60",
    "abs_pre_ret_2","abs_pre_ret_5","abs_pre_ret_15","abs_pre_ret_30","abs_pre_ret_60",
    "entry_bar_ret","entry_bar_range","range_5m","range_30m","close_position_60",
    "volume_ratio_5_30","pre_vol_10","log_entry_price","excess_ret_30",
]


def rank(value):
    x=np.asarray(value,dtype=float);finite=x[np.isfinite(x)]
    fill=float(np.median(finite)) if len(finite) else 0.0
    x=np.nan_to_num(x,nan=fill,posinf=fill,neginf=fill);ref=np.sort(x)
    return np.searchsorted(ref,x,side="right")/len(ref)


def base(dev,market,source):
    target=dev[dev.market.eq(market)&dev.source.eq(source)&dev.y.notna()].copy().reset_index(drop=True)
    opened=dev[dev.market.eq(market)&~dev.source.eq(source)&dev.y.notna()].copy()
    prior=opened[opened.ticker.astype(str).isin(set(target.ticker.astype(str)))].copy()
    model=app.make_model(MODEL[market]["spec"]);model.fit(app.model_frame(prior),prior.y.astype(int))
    target["raw_prob"]=model.predict_proba(app.model_frame(target))[:,1]
    target["direction_margin"]=rank(np.abs(target.raw_prob-MODEL[market]["cutoff"]))
    target["audit2"]=target.event_id.astype(str).map(lambda v:int(hashlib.sha256(f"V18-AUDIT|{v}".encode()).hexdigest(),16)%2)
    target["audit3"]=target.event_id.astype(str).map(lambda v:int(hashlib.sha256(f"V18-THIRD|{v}".encode()).hexdigest(),16)%3)
    features=app.model_frame(target)
    for column in OPPORTUNITY:target[column]=pd.to_numeric(features[column],errors="coerce").to_numpy(float)
    return target


def gate(raw,market,column,sign,weight,quantile):
    opportunity=rank(sign*raw[column].to_numpy(float));margin=raw.direction_margin.to_numpy(float)
    score=rank(weight*margin+(1-weight)*opportunity);high=score>=quantile
    confidence=np.where(high,.86+.14*margin,.84*margin);cutoff=MODEL[market]["cutoff"]
    probability=np.where(raw.raw_prob>=cutoff,.5+.5*confidence,.5-.5*confidence)
    out=raw.copy();out["prob"]=probability
    return out


def halves(frame):
    yield "all",frame
    for value in (0,1):yield f"half{value}",frame[frame.audit2.eq(value)].copy()


def all_slices(frame):
    yield from halves(frame)
    for value in (0,1,2):yield f"third{value}",frame[frame.audit3.eq(value)].copy()


def main():
    cfg=app.Config();data=app.load_search_frame(cfg);dev,_,_=app.split_dev_seal(data,cfg)
    raw={m:base(dev,m,s) for m,s in (("US","SEC_V18_DEV"),("KR","KIND_V18_DEV"))}
    candidates={}
    for market in ("US","KR"):
        rows=[]
        for column,sign,weight,q in itertools.product(OPPORTUNITY,(-1,1),(0,.25,.5,.75,1),(.65,.70,.75,.80,.85)):
            part=gate(raw[market],market,column,sign,weight,q);mets={name:app.evaluate(x,.925,.002) for name,x in halves(part)}
            score=(3*min(x["highconf_accuracy"] for x in mets.values())+
                   30*min(x["strategy_mean_signed_net"] for x in mets.values())+
                   .2*min(x["highconf_coverage"] for x in mets.values()))
            rows.append({"column":column,"sign":sign,"weight":weight,"gate":q,"score":score,"metrics":mets})
        rows.sort(key=lambda x:x["score"],reverse=True);candidates[market]=rows[:30]
        print(json.dumps({"market":market,"top":rows[:10]},indent=2))
    pairs=[]
    for us in candidates["US"]:
        up=gate(raw["US"],"US",us["column"],us["sign"],us["weight"],us["gate"])
        for kr in candidates["KR"]:
            kp=gate(raw["KR"],"KR",kr["column"],kr["sign"],kr["weight"],kr["gate"])
            full=pd.concat([up,kp],ignore_index=True);mets={}
            for name,x in all_slices(full):
                x=x.copy()
                for market in ("US","KR"):
                    mask=x.market.eq(market);x.loc[mask,"eval_weight"]=TARGET[market]/mask.sum()
                mets[name]=app.evaluate(x,.925,.002)
            margins=[]
            for value in mets.values():
                margins += [value["balanced_accuracy"]-.54,value["auc"]-.56,
                            value["edge_vs_naive"]-.02,
                            value["highconf_coverage"]-.15,
                            value["highconf_accuracy"]-.60,
                            20*value["strategy_mean_signed_net"],
                            value["by_market"]["US"]["balanced_accuracy"]-.51,
                            value["by_market"]["KR"]["balanced_accuracy"]-.51]
            pairs.append({"us":{k:us[k] for k in ("column","sign","weight","gate")},
                          "kr":{k:kr[k] for k in ("column","sign","weight","gate")},
                          "pass_count":sum(v>=0 for v in margins),"min_margin":min(margins),"metrics":mets})
    pairs.sort(key=lambda x:(x["pass_count"],x["min_margin"]),reverse=True)
    print(json.dumps({"pairs":pairs[:20]},indent=2))


if __name__=="__main__":main()
