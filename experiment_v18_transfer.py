"""Audit the selected V18 policy on already-opened V17 seal outcomes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY={
    "US":{"components":[("entry_bar_ret",-1),("excess_ret_5",-1)],"cutoff":.39,
          "opportunity":"abs_pre_ret_2","opportunity_sign":1,"margin_weight":.5,"gate":.75},
    "KR":{"components":[("close_position_60",1),("log_entry_price",1),("benchmark_ret_60",1)],"cutoff":.55,
          "opportunity":"pre_ret_10","opportunity_sign":1,"margin_weight":.5,"gate":.75},
}


def numeric(frame,column,median=None):
    value=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
    if median is None:median=float(np.nanmedian(value)) if np.isfinite(value).any() else 0.0
    return np.nan_to_num(value,nan=median,posinf=median,neginf=median),median


def rank(ref,value):
    return np.searchsorted(np.sort(ref),value,side="right")/len(ref)


def fit_predict(train,test,market):
    spec=POLICY[market];tr=app.model_frame(train);te=app.model_frame(test);pieces=[];refs={};med={}
    for column,sign in spec["components"]:
        a,med[column]=numeric(tr,column);b,_=numeric(te,column,med[column]);refs[column]=np.sort(a)
        component=rank(refs[column],b);pieces.append(component if sign>0 else 1-component)
    direction=np.mean(pieces,axis=0)
    # Fit confidence references on the V18 development data only.
    train_pieces=[]
    for column,sign in spec["components"]:
        a,_=numeric(tr,column,med[column]);component=rank(refs[column],a)
        train_pieces.append(component if sign>0 else 1-component)
    train_direction=np.mean(train_pieces,axis=0)
    train_margin_raw=np.abs(train_direction-spec["cutoff"]);margin_ref=np.sort(train_margin_raw)
    margin=rank(margin_ref,np.abs(direction-spec["cutoff"]));train_margin=rank(margin_ref,train_margin_raw)
    a,opp_med=numeric(tr,spec["opportunity"]);b,_=numeric(te,spec["opportunity"],opp_med)
    a=spec["opportunity_sign"]*a;b=spec["opportunity_sign"]*b;opp_ref=np.sort(a)
    train_opp=rank(opp_ref,a);opp=rank(opp_ref,b)
    train_raw=spec["margin_weight"]*train_margin+(1-spec["margin_weight"])*train_opp
    conf_ref=np.sort(train_raw);score=rank(conf_ref,spec["margin_weight"]*margin+(1-spec["margin_weight"])*opp)
    confidence=np.where(score>=spec["gate"],.86+.14*margin,.84*margin)
    probability=np.where(direction>=spec["cutoff"],.5+.5*confidence,.5-.5*confidence)
    out=test.copy();out["prob"]=probability
    return out


def main():
    data=app.load_search_frame(app.Config());parts=[]
    for market,dev_source,audit_source in (("US","SEC_V18_DEV","SEC_V17_SEAL"),("KR","KIND_V18_DEV","KIND_V17_SEAL")):
        train=data[data.market.eq(market)&data.source.eq(dev_source)&data.y.notna()].copy()
        test=data[data.market.eq(market)&data.source.eq(audit_source)&data.y.notna()].copy()
        result=fit_predict(train,test,market);parts.append(result)
        print(json.dumps({"market":market,"metrics":app.evaluate(result,.925,.002)},indent=2))
    full=pd.concat(parts,ignore_index=True)
    print(json.dumps({"combined":app.evaluate(full,.925,.002)},indent=2))


if __name__=="__main__":main()
