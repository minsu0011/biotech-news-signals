"""Search a robust high-confidence gate after freezing V18 DEV direction."""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


POLICY={
    "US":{"components":[("entry_bar_ret",-1),("excess_ret_5",-1)],"cutoff":.39},
    "KR":{"components":[("close_position_60",1),("log_entry_price",1),("benchmark_ret_60",1)],"cutoff":.55},
}
TARGET={"US":395.0,"KR":633.0}


def rank(value: np.ndarray) -> np.ndarray:
    x=np.asarray(value,dtype=float);finite=x[np.isfinite(x)]
    fill=float(np.median(finite)) if len(finite) else 0.0
    x=np.nan_to_num(x,nan=fill,posinf=fill,neginf=fill);ref=np.sort(x)
    return np.searchsorted(ref,x,side="right")/len(ref)


def base(raw: pd.DataFrame,market: str) -> pd.DataFrame:
    frame=app.model_frame(raw);spec=POLICY[market];pieces=[]
    for column,sign in spec["components"]:
        component=rank(pd.to_numeric(frame[column],errors="coerce").to_numpy(float))
        pieces.append(component if sign>0 else 1.0-component)
    out=raw.copy();out["direction"]=np.mean(pieces,axis=0)
    out["margin"]=rank(np.abs(out.direction.to_numpy()-spec["cutoff"]))
    out["audit2"]=out.event_id.astype(str).map(lambda v:int(hashlib.sha256(f"V18-AUDIT|{v}".encode()).hexdigest(),16)%2)
    out["audit3"]=out.event_id.astype(str).map(lambda v:int(hashlib.sha256(f"V18-THIRD|{v}".encode()).hexdigest(),16)%3)
    for column in app.NUMERIC_COLS:
        out[column]=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
    return out


def apply_gate(raw: pd.DataFrame,market: str,column: str,sign: int,
               margin_weight: float,gate: float) -> pd.DataFrame:
    opportunity=rank(sign*raw[column].to_numpy(float))
    score=rank(margin_weight*raw.margin.to_numpy()+(1-margin_weight)*opportunity)
    high=score>=gate;confidence=np.where(high,.86+.14*raw.margin,.84*raw.margin)
    cutoff=POLICY[market]["cutoff"]
    probability=np.where(raw.direction>=cutoff,.5+.5*confidence,.5-.5*confidence)
    out=raw.copy();out["prob"]=probability
    return out


def slices(frame: pd.DataFrame):
    yield "all",frame
    for value in (0,1):yield f"half{value}",frame[frame.audit2.eq(value)].copy()
    for value in (0,1,2):yield f"third{value}",frame[frame.audit3.eq(value)].copy()


def main() -> None:
    data=app.load_search_frame(app.Config());dev,_,_=app.split_dev_seal(data,app.Config())
    raw={
        market:base(dev[dev.market.eq(market)&dev.source.eq(source)&dev.y.notna()].reset_index(drop=True),market)
        for market,source in (("US","SEC_V18_DEV"),("KR","KIND_V18_DEV"))
    }
    candidates={}
    for market in ("US","KR"):
        rows=[]
        for column,sign,weight,gate in itertools.product(
                app.NUMERIC_COLS,(-1,1),(0,.25,.50,.75,1.0),(.50,.55,.60,.65,.70,.75,.80,.85)):
            part=apply_gate(raw[market],market,column,sign,weight,gate)
            mets={name:app.evaluate(x,.925,.002) for name,x in slices(part)}
            min_acc=min(value["highconf_accuracy"] for value in mets.values())
            min_net=min(value["strategy_mean_signed_net"] for value in mets.values())
            min_cov=min(value["highconf_coverage"] for value in mets.values())
            score=3*min_acc+30*min_net+.25*min_cov+mets["all"]["highconf_accuracy"]
            rows.append({"column":column,"sign":sign,"margin_weight":weight,"gate":gate,
                         "score":score,"min_acc":min_acc,"min_net":min_net,
                         "min_cov":min_cov,"metrics":mets})
        rows.sort(key=lambda x:x["score"],reverse=True);candidates[market]=rows[:40]
        print(json.dumps({"market":market,"top":rows[:15]},indent=2))

    pairs=[]
    for us in candidates["US"]:
        up=apply_gate(raw["US"],"US",us["column"],us["sign"],us["margin_weight"],us["gate"])
        for kr in candidates["KR"]:
            kp=apply_gate(raw["KR"],"KR",kr["column"],kr["sign"],kr["margin_weight"],kr["gate"])
            full=pd.concat([up,kp],ignore_index=True);mets={}
            for name,x in slices(full):
                x=x.copy()
                for market in ("US","KR"):
                    mask=x.market.eq(market);x.loc[mask,"eval_weight"]=TARGET[market]/mask.sum()
                mets[name]=app.evaluate(x,.925,.002)
            margins=[]
            for value in mets.values():
                margins += [value["highconf_coverage"]-.15,
                            value["highconf_accuracy"]-.60,
                            20*value["strategy_mean_signed_net"]]
            pairs.append({"us":{k:us[k] for k in ("column","sign","margin_weight","gate")},
                          "kr":{k:kr[k] for k in ("column","sign","margin_weight","gate")},
                          "pass_count":sum(x>=0 for x in margins),"min_margin":min(margins),
                          "metrics":mets})
    pairs.sort(key=lambda x:(x["pass_count"],x["min_margin"]),reverse=True)
    print(json.dumps({"pairs":pairs[:25]},indent=2))


if __name__=="__main__":main()
