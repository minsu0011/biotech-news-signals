"""V20 DEV-only rank direction and high-confidence policy search."""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def bucket(value:str,modulo:int)->int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16],16)%modulo


def rank(values:pd.Series)->np.ndarray:
    x=pd.to_numeric(values,errors="coerce")
    fill=float(x.median()) if x.notna().any() else 0.0
    return x.fillna(fill).rank(method="average",pct=True).to_numpy(float)


def direction(frame:pd.DataFrame,column:str,sign:float)->np.ndarray:
    score=rank(frame[column]);return score if sign>0 else 1.0-score


def subset_metric(frame:pd.DataFrame,score:np.ndarray,cutoff:float,selected:np.ndarray|None=None)->dict:
    y=frame.y.astype(int).to_numpy();pred=(score>=cutoff).astype(int)
    if selected is None:selected=np.ones(len(frame),dtype=bool)
    signed=np.where(pred[selected]==1,1.0,-1.0)*frame.fwd_ret_30m.to_numpy(float)[selected]
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    return {"n":int(selected.sum()),"coverage":float(selected.mean()),
            "acc":float((pred[selected]==y[selected]).mean()),
            "bal":float(balanced_accuracy_score(y,pred)),
            "auc":float(roc_auc_score(y,score)),
            "net":float(signed.mean()-0.002)}


def masks(frame:pd.DataFrame)->dict[str,np.ndarray]:
    event=frame.event_id.astype(str);h=event.map(lambda value:bucket(value,2)).to_numpy()
    time=pd.to_datetime(frame.event_time_utc,utc=True).rank(method="first",pct=True).to_numpy()
    return {"all":np.ones(len(frame),dtype=bool),"h0":h==0,"h1":h==1,
            "early":time<=0.5,"late":time>0.5}


def gate_grid(frame:pd.DataFrame,column:str,sign:float,cutoff:float,top:int=100)->list[dict]:
    score=direction(frame,column,sign);pred=(score>=cutoff).astype(int)
    margin=rank(pd.Series(np.abs(score-cutoff)))
    opportunities=(
        "abs_pre_ret_2","abs_pre_ret_5","abs_pre_ret_15","abs_pre_ret_30","abs_pre_ret_60",
        "pre_vol_5","pre_vol_10","pre_vol_30","pre_vol_60","range_5m","range_10m",
        "range_30m","range_60m","volume_ratio_1_30","volume_ratio_2_30",
        "volume_ratio_5_30","volume_ratio_10_60","entry_bar_range","minutes_to_close",
        "benchmark_ret_2","benchmark_ret_5","benchmark_ret_15","benchmark_ret_30",
        "excess_ret_5","excess_ret_15","excess_ret_30",
    )
    y=frame.y.astype(int).to_numpy();returns=frame.fwd_ret_30m.to_numpy(float);ms=masks(frame)
    rows=[]
    for opportunity_column in opportunities:
        base=pd.to_numeric(frame[opportunity_column],errors="coerce")
        for opportunity_mode in ("high","low"):
            opportunity=rank(base)
            if opportunity_mode=="low":opportunity=1.0-opportunity
            for margin_weight in (0.0,0.25,0.5,0.75,1.0):
                confidence=margin_weight*margin+(1.0-margin_weight)*opportunity
                for quantile in (0.60,0.65,0.70,0.75,0.80,0.825,0.85):
                    selected=confidence>=np.quantile(confidence,quantile)
                    audit={}
                    for name,mask in ms.items():
                        use=selected&mask
                        if use.sum()<8:continue
                        signed=np.where(pred[use]==1,1.0,-1.0)*returns[use]
                        audit[name]={"n":int(use.sum()),"coverage":float(use.sum()/mask.sum()),
                                     "acc":float((pred[use]==y[use]).mean()),
                                     "net":float(signed.mean()-0.002)}
                    if not set(ms).issubset(audit):continue
                    core=list(audit.values())
                    value=(3*audit["all"]["acc"]+50*audit["all"]["net"]+
                           min(x["acc"] for x in core)+20*min(x["net"] for x in core)+
                           0.2*audit["all"]["coverage"])
                    rows.append({"opportunity_column":opportunity_column,
                                 "opportunity_mode":opportunity_mode,
                                 "margin_weight":margin_weight,"gate_quantile":quantile,
                                 "value":float(value),"audits":audit})
    return sorted(rows,key=lambda value:value["value"],reverse=True)[:top]


def prior_audits(data:pd.DataFrame,target:pd.DataFrame,market:str,column:str,sign:float,cutoff:float)->dict:
    groups={}
    if market=="US":
        tickers=set(target.ticker.astype(str))
        for source in ("SEC_V17_DEV","SEC_V17_SEAL","SEC_V18_DEV","SEC_V18_SEAL",
                       "SEC_V19_DEV","SEC_V19_SEAL"):
            part=data[data.source.eq(source)&data.y.notna()&data.ticker.astype(str).isin(tickers)].copy()
            if len(part)>=30:groups[source]=part
    else:
        for version in ("V12","V13","V14"):
            part=data[data.source.astype(str).str.startswith(f"NAVER_NEWS_{version}")&data.y.notna()].copy()
            if len(part)>=30:groups[version]=part
    out={}
    for name,raw in groups.items():
        frame=app.model_frame(raw);frame[["y","fwd_ret_30m"]]=raw[["y","fwd_ret_30m"]]
        out[name]=subset_metric(frame,direction(frame,column,sign),cutoff)
    return out


def encode(frame:pd.DataFrame,column:str,sign:float,cutoff:float,gate:dict)->tuple[np.ndarray,np.ndarray]:
    score=direction(frame,column,sign);up=score>=cutoff
    margin=rank(pd.Series(np.abs(score-cutoff)))
    opportunity=rank(pd.to_numeric(frame[gate["opportunity_column"]],errors="coerce"))
    if gate["opportunity_mode"]=="low":opportunity=1.0-opportunity
    confidence_raw=gate["margin_weight"]*margin+(1.0-gate["margin_weight"])*opportunity
    high=confidence_raw>=np.quantile(confidence_raw,gate["gate_quantile"])
    confidence=np.where(high,0.86+0.14*margin,0.84*margin)
    probability=np.where(up,0.5+0.5*confidence,0.5-0.5*confidence)
    return probability,high


def main()->None:
    cfg=app.Config();data=app.load_search_frame(cfg)
    assert data.loc[data.source.isin({"SEC_V20_SEAL","NAVER_NEWS_V20_SEAL"}),"y"].isna().all()
    policy={"US":("SEC_V20_DEV","volume_ratio_1_30",-1.0,0.525,346.0),
            "KR":("NAVER_NEWS_V20_DEV","benchmark_ret_15",-1.0,0.575,522.0)}
    frames={};output={"seal_outcomes_loaded":False,"markets":{}}
    for market,(source,column,sign,cutoff,target_weight) in policy.items():
        raw=data[data.source.eq(source)&data.y.notna()].copy().reset_index(drop=True)
        frame=app.model_frame(raw);frame[["y","fwd_ret_30m"]]=raw[["y","fwd_ret_30m"]]
        frames[market]=frame
        output["markets"][market]={
            "direction":subset_metric(frame,direction(frame,column,sign),cutoff),
            "prior_audits":prior_audits(data,raw,market,column,sign,cutoff),
            "gates":gate_grid(frame,column,sign,cutoff),
        }
    # Pair the leading robust gates and evaluate the exact weighted certificate metrics.
    pairs=[]
    for us_gate,kr_gate in itertools.product(output["markets"]["US"]["gates"][:30],
                                             output["markets"]["KR"]["gates"][:30]):
        parts=[]
        for market,gate in (("US",us_gate),("KR",kr_gate)):
            _,column,sign,cutoff,target_weight=policy[market];frame=frames[market]
            probability,_=encode(frame,column,sign,cutoff,gate)
            part=frame[["event_id","event_time_utc","y","fwd_ret_30m"]].copy()
            part["market"]=market;part["prob"]=probability;part["eval_weight"]=target_weight/len(part)
            parts.append(part)
        combined=pd.concat(parts,ignore_index=True)
        metrics=app.evaluate(combined,0.925,0.002)
        value=(3*metrics["balanced_accuracy"]+2*metrics["auc"]+
               3*metrics["highconf_accuracy"]+40*metrics["strategy_mean_signed_net"]+
               0.3*metrics["highconf_coverage"])
        pairs.append({"value":float(value),"us_gate":us_gate,"kr_gate":kr_gate,"metrics":metrics})
    output["pairs"]=sorted(pairs,key=lambda value:value["value"],reverse=True)[:50]
    print(json.dumps(output,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
