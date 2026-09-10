"""V21 robust US rule, KR ensemble, and confidence-gate selection."""
from __future__ import annotations

import runtime_limits

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

import bio_news_30m_v3 as app
import experiment_v21_models as scan


def rank(values:pd.Series)->np.ndarray:
    x=pd.to_numeric(values,errors="coerce");fill=float(x.median()) if x.notna().any() else 0.
    return x.fillna(fill).rank(method="average",pct=True).to_numpy(float)


def score(frame:pd.DataFrame,components:tuple[tuple[str,float],...])->np.ndarray:
    pieces=[]
    features=app.model_frame(frame)
    for column,sign in components:
        value=rank(features[column]);pieces.append(value if sign>0 else 1-value)
    return np.mean(pieces,axis=0)


def basic(frame:pd.DataFrame,probability:np.ndarray,cutoff:float)->dict:
    y=frame.y.astype(int).to_numpy();pred=(probability>=cutoff).astype(int)
    return {"n":len(frame),"bal":float(balanced_accuracy_score(y,pred)),
            "auc":float(roc_auc_score(y,probability)),"acc":float((y==pred).mean())}


def us_search(data:pd.DataFrame,target:pd.DataFrame)->list[dict]:
    audits={"v21_dev":target,
            "v20_dev":data[data.source.eq("SEC_V20_DEV")&data.y.notna()].copy(),
            "v20_seal":data[data.source.eq("SEC_V20_SEAL")&data.y.notna()].copy()}
    atoms=(("benchmark_ret_60",1.),("trend_slope_30",1.),("up_fraction_10",1.),
           ("ret_slope_2_15",-1.),("log_entry_price",1.),("benchmark_ret_30",1.))
    rows=[]
    for size in (1,2,3):
        for components in itertools.combinations(atoms,size):
            scores={name:score(frame,components) for name,frame in audits.items()}
            for cutoff in np.arange(.4,.601,.025):
                result={name:basic(audits[name],scores[name],float(cutoff)) for name in audits}
                core=list(result.values());value=2*min(x['bal'] for x in core)+min(x['auc'] for x in core)+result['v21_dev']['bal']+result['v21_dev']['auc']
                rows.append({"components":components,"cutoff":float(cutoff),"value":value,"audits":result})
    return sorted(rows,key=lambda value:value['value'],reverse=True)


def confidence_gate(frame:pd.DataFrame,direction_probability:np.ndarray,cutoff:float,top:int=80)->list[dict]:
    pred=(direction_probability>=cutoff).astype(int);y=frame.y.astype(int).to_numpy();ret=frame.fwd_ret_30m.to_numpy(float)
    margin=rank(pd.Series(np.abs(direction_probability-cutoff)));features=app.model_frame(frame)
    event=frame.event_id.astype(str);h=event.map(lambda x:int(hashlib.sha256(x.encode()).hexdigest()[:16],16)%2).to_numpy()
    time=pd.to_datetime(frame.event_time_utc,utc=True).rank(method="first",pct=True).to_numpy()
    ticker=frame.ticker.astype(str).map(lambda x:int(hashlib.sha256(x.encode()).hexdigest()[:16],16)%2).to_numpy()
    masks={"all":np.ones(len(frame),bool),"h0":h==0,"h1":h==1,"t0":time<=.5,"t1":time>.5,"ticker0":ticker==0,"ticker1":ticker==1}
    columns=("abs_pre_ret_2","abs_pre_ret_5","abs_pre_ret_15","abs_pre_ret_30","pre_vol_10","pre_vol_30",
             "range_5m","range_10m","range_30m","volume_ratio_1_30","volume_ratio_2_30","volume_ratio_5_30",
             "volume_ratio_10_60","entry_bar_range","minutes_to_close","pre_ret_5","pre_ret_15","pre_ret_30")
    rows=[]
    for column in columns:
        raw=pd.to_numeric(features[column],errors="coerce")
        for mode in ("high","low"):
            opportunity=rank(raw);opportunity=opportunity if mode=="high" else 1-opportunity
            for weight in (0.,.25,.5,.75,1.):
                conf=weight*margin+(1-weight)*opportunity
                for quantile in (.6,.65,.7,.75,.775,.8,.825,.85):
                    selected=conf>=np.quantile(conf,quantile);result={}
                    for name,mask in masks.items():
                        use=selected&mask
                        if use.sum()<20:continue
                        signed=np.where(pred[use]==1,1.,-1.)*ret[use]-.002
                        result[name]={"n":int(use.sum()),"coverage":float(use.sum()/mask.sum()),
                                      "acc":float((pred[use]==y[use]).mean()),"net":float(signed.mean())}
                    if not set(masks).issubset(result):continue
                    core=list(result.values());value=3*result['all']['acc']+50*result['all']['net']+min(x['acc'] for x in core)+20*min(x['net'] for x in core)+.2*result['all']['coverage']
                    rows.append({"column":column,"mode":mode,"margin_weight":weight,"quantile":quantile,"value":value,"audits":result})
    return sorted(rows,key=lambda value:value['value'],reverse=True)[:top]


def encode(probability:np.ndarray,cutoff:float,frame:pd.DataFrame,gate:dict|None)->np.ndarray:
    up=probability>=cutoff;margin=rank(pd.Series(np.abs(probability-cutoff)));high=np.zeros(len(frame),bool)
    if gate is not None:
        features=app.model_frame(frame);opportunity=rank(pd.to_numeric(features[gate['column']],errors='coerce'))
        if gate['mode']=='low':opportunity=1-opportunity
        raw=gate['margin_weight']*margin+(1-gate['margin_weight'])*opportunity
        high=raw>=np.quantile(raw,gate['quantile'])
    confidence=np.where(high,.86+.14*margin,.84*margin)
    return np.where(up,.5+.5*confidence,.5-.5*confidence)


def main()->None:
    data=app.load_search_frame(app.Config());assert data.loc[data.source.isin({'SEC_V21_SEAL','NAVER_NEWS_V21_SEAL'}),'y'].isna().all()
    us=data[data.source.eq('SEC_V21_DEV')&data.y.notna()].copy().reset_index(drop=True)
    kr=data[data.source.eq('NAVER_NEWS_V21_DEV')&data.y.notna()].copy().reset_index(drop=True)
    us_rows=us_search(data,us);us_win=us_rows[0];us_probability=score(us,tuple(tuple(x) for x in us_win['components']))
    kr_specs=[{"kind":"native_cat","iterations":220,"depth":d,"cat_cols":["event_type","ticker"]} for d in (2,4,5)]
    kr_raw=[scan.crossfit(kr,spec,'KR') for spec in kr_specs]
    ensembles=[]
    for weights in ((0,1,0),(1,1,0),(0,1,1),(1,2,1),(1,1,1)):
        probability=np.average(np.stack(kr_raw),axis=0,weights=weights);value,cutoff,audits=scan.select_cutoff(kr,probability)
        ensembles.append({"weights":weights,"probability":probability,"value":value,"cutoff":cutoff,"audits":audits})
    kr_win=max(ensembles,key=lambda x:x['value']);kr_gates=confidence_gate(kr,kr_win['probability'],kr_win['cutoff'])
    pairs=[]
    for gate in kr_gates[:40]:
        up=encode(us_probability,us_win['cutoff'],us,None);kp=encode(kr_win['probability'],kr_win['cutoff'],kr,gate)
        parts=[]
        for market,frame,prob,target_weight in (("US",us,up,342.),("KR",kr,kp,1342.)):
            z=frame[["event_id","event_time_utc","y","fwd_ret_30m"]].copy();z['market']=market;z['prob']=prob;z['eval_weight']=target_weight/len(z);parts.append(z)
        combined=pd.concat(parts,ignore_index=True);metrics=app.evaluate(combined,.925,.002)
        value=3*metrics['balanced_accuracy']+2*metrics['auc']+3*metrics['highconf_accuracy']+40*metrics['strategy_mean_signed_net']+.3*metrics['highconf_coverage']
        pairs.append({"value":value,"gate":gate,"metrics":metrics})
    result={"seal_outcomes_loaded":False,"us_top":us_rows[:20],"kr_ensembles":[{k:v for k,v in x.items() if k!='probability'} for x in ensembles],
            "kr_gates":kr_gates,"pairs":sorted(pairs,key=lambda x:x['value'],reverse=True)}
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__":main()
