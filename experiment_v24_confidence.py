"""V24 confidence gates on fixed strong direction plateaus."""
from __future__ import annotations

import runtime_limits

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
from experiment_v24_direction import ENSEMBLES,TARGET,ensemble


ROOT=Path(__file__).resolve().parent
DIRECTIONS={
    "plateau":{"US":{"ensemble":"u_all","cutoff":.540},
               "KR":{"ensemble":"k_cat_lgb","cutoff":.585}},
    "plateau_hi":{"US":{"ensemble":"u_all","cutoff":.545},
                  "KR":{"ensemble":"k_cat_lgb","cutoff":.590}},
    "bootstrap":{"US":{"ensemble":"u_lgb5","cutoff":.385},
                 "KR":{"ensemble":"k_cat_lgb","cutoff":.585}},
    "single":{"US":{"ensemble":"u_lgb5","cutoff":.385},
              "KR":{"ensemble":"k_cat6","cutoff":.590}},
}
GATE_COLUMNS=(
    "volume_ratio_10_60","volume_ratio_5_30","volume_ratio_2_30",
    "pre_vol_10","pre_vol_30","abs_pre_ret_5","abs_pre_ret_15","abs_pre_ret_30",
    "range_10m","range_30m","entry_bar_range","benchmark_ret_5","benchmark_ret_30",
    "minutes_to_close","pre_ret_2","pre_ret_5","pre_ret_15","pre_ret_30",
    "excess_ret_5","excess_ret_15","excess_ret_30",
)


def rank(values:np.ndarray)->np.ndarray:
    return pd.Series(values).rank(method="average",pct=True).to_numpy(float)


def masks(frame:pd.DataFrame)->dict[str,np.ndarray]:
    event_hash=frame.event_id.astype(str).map(
        lambda value:int(hashlib.sha256(value.encode()).hexdigest()[:16],16)%2).to_numpy()
    ticker_hash=frame.ticker.astype(str).map(
        lambda value:int(hashlib.sha256(value.encode()).hexdigest()[:16],16)%2).to_numpy()
    time_rank=pd.to_datetime(frame.event_time_utc,utc=True).rank(method="first",pct=True).to_numpy()
    return {"all":np.ones(len(frame),bool),"h0":event_hash==0,"h1":event_hash==1,
            "t0":time_rank<=.5,"t1":time_rank>.5,
            "ticker0":ticker_hash==0,"ticker1":ticker_hash==1}


def encode(feature:pd.DataFrame,raw:np.ndarray,cutoff:float,gate:dict)->np.ndarray:
    margin=rank(np.abs(raw-cutoff))
    value=pd.to_numeric(feature[gate["column"]],errors="coerce")
    fill=float(value.median()) if value.notna().any() else 0.0
    opportunity=rank(value.fillna(fill).to_numpy(float))
    if gate["sign"]<0:opportunity=1-opportunity
    confidence_raw=gate["margin_weight"]*margin+(1-gate["margin_weight"])*opportunity
    selected=rank(confidence_raw)>=gate["gate_quantile"]
    confidence=np.where(selected,.86+.14*margin,.84*margin)
    return np.where(raw>=cutoff,.5+.5*confidence,.5-.5*confidence)


def candidates()->list[dict]:
    return [{"column":column,"sign":sign,"margin_weight":weight,"gate_quantile":quantile}
            for column,sign,weight,quantile in itertools.product(
                GATE_COLUMNS,(-1,1),(.20,.35,.50,.75,1.0),(.65,.70,.75,.80,.825,.85,.875,.90))]


def shortlist(frame:pd.DataFrame,feature:pd.DataFrame,raw:np.ndarray,cutoff:float)->list[dict]:
    y=frame.y.astype(int).to_numpy();ret=frame.fwd_ret_30m.to_numpy(float)
    pred=raw>=cutoff;split_masks=masks(frame);rows=[]
    for gate in candidates():
        probability=encode(feature,raw,cutoff,gate)
        selected=np.maximum(probability,1-probability)>=.925;split={}
        for name,mask in split_masks.items():
            use=selected&mask
            if use.sum()<5:break
            net=np.where(pred[use],1.,-1.)*ret[use]-.002
            split[name]={"n":int(use.sum()),"coverage":float(use.sum()/mask.sum()),
                         "accuracy":float((pred[use]==y[use]).mean()),"net":float(net.mean())}
        if len(split)!=len(split_masks):continue
        full=split["all"]
        score=(5*full["accuracy"]+100*full["net"]+full["coverage"]+
               min(v["accuracy"] for v in split.values())+
               30*min(v["net"] for v in split.values()))
        rows.append({"gate":gate,"score":float(score),"split":split})
    return sorted(rows,key=lambda row:row["score"],reverse=True)[:35]


def eval_frame(frame:pd.DataFrame,probability:np.ndarray,market:str)->pd.DataFrame:
    z=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    z["prob"]=probability;z["eval_weight"]=TARGET[market]/len(z);return z


def main()->None:
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V24_SEAL","NAVER_NEWS_V24_SEAL"}),"y"].notna().any():
        raise RuntimeError("V24 seal outcome exposure detected")
    raw_cache=pd.read_parquet(ROOT/"cache"/"v24_model_oof.parquet")
    frames={
        "US":data[data.source.eq("SEC_V24_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),
        "KR":data[data.source.eq("NAVER_NEWS_V24_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),
    }
    features={market:app.model_frame(frame) for market,frame in frames.items()}
    wide={market:(raw_cache[raw_cache.market.eq(market)]
                  .pivot(index="event_id",columns="model_name",values="probability")
                  .reindex(frames[market].event_id)) for market in frames}
    raw={market:{name:ensemble(wide[market],weights) for name,weights in ENSEMBLES[market].items()}
         for market in frames}
    results=[];gate_audits={}
    for direction_name,direction in DIRECTIONS.items():
        gates={}
        for market in ("US","KR"):
            choice=direction[market]
            gates[market]=shortlist(frames[market],features[market],
                                     raw[market][choice["ensemble"]],choice["cutoff"])
        gate_audits[direction_name]=gates
        for us_gate,kr_gate in itertools.product(gates["US"],gates["KR"]):
            parts=[]
            for market,gate in (("US",us_gate),("KR",kr_gate)):
                choice=direction[market]
                probability=encode(features[market],raw[market][choice["ensemble"]],
                                   choice["cutoff"],gate["gate"])
                parts.append(eval_frame(frames[market],probability,market))
            combined=pd.concat(parts,ignore_index=True);metric=app.evaluate(combined,.925,.002)
            required=(metric["balanced_accuracy"]>=.54,metric["auc"]>=.56,
                      metric["edge_vs_naive"]>=.02,metric["highconf_coverage"]>=.15,
                      metric["highconf_accuracy"]>=.60,metric["strategy_mean_signed_net"]>0,
                      metric["by_market"]["US"]["balanced_accuracy"]>=.51,
                      metric["by_market"]["KR"]["balanced_accuracy"]>=.51)
            score=(4*metric["balanced_accuracy"]+2*metric["auc"]+
                   3*metric["highconf_accuracy"]+50*metric["strategy_mean_signed_net"]+
                   metric["edge_vs_naive"]+.2*metric["highconf_coverage"])
            results.append({"direction_name":direction_name,"direction":direction,
                            "us_gate":us_gate,"kr_gate":kr_gate,
                            "pass_count":int(sum(required)),"score":float(score),"metrics":metric})
    results.sort(key=lambda row:(row["pass_count"],row["score"]),reverse=True)
    for row in results[:50]:
        parts=[]
        for market,key in (("US","us_gate"),("KR","kr_gate")):
            choice=row["direction"][market]
            probability=encode(features[market],raw[market][choice["ensemble"]],
                               choice["cutoff"],row[key]["gate"])
            parts.append(eval_frame(frames[market],probability,market))
        row["bootstrap"]=app.bootstrap_by_day(pd.concat(parts,ignore_index=True),.925,.002,nboot=1000)
        row["pass_count"]+=int(row["bootstrap"]["balanced_accuracy_lower95"]>=.50)
    results[:50]=sorted(results[:50],key=lambda row:(row["pass_count"],row["score"]),reverse=True)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),
            "directions":DIRECTIONS,"gate_audits":gate_audits,"pairs":results[:120]}
    path=ROOT/"cache"/"v24_confidence_audits.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for row in results[:20]:
        metric=row["metrics"]
        print(f"passes={row['pass_count']}/9 direction={row['direction_name']} "
              f"bal={metric['balanced_accuracy']:.4f} auc={metric['auc']:.4f} "
              f"edge={metric['edge_vs_naive']:.4f} "
              f"hc={metric['highconf_accuracy']:.4f}/{metric['highconf_coverage']:.4f} "
              f"net={metric['strategy_mean_signed_net']:.5f} "
              f"boot={row.get('bootstrap',{}).get('balanced_accuracy_lower95')}",flush=True)
    print("RESULT="+str(path))


if __name__=="__main__":main()
