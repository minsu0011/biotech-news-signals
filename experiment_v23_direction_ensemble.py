"""V23 augmented-OOF direction ensemble and cutoff pairing."""
from __future__ import annotations

import runtime_limits

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

import bio_news_30m_v3 as app


ROOT=Path(__file__).resolve().parent
TARGET={"US":256.0,"KR":365.0}
ENSEMBLES={
    "US":{
        "u_lgb":{"lgb9_r0":1},
        "u_lgb_cat2":{"lgb9_r0":2,"cat2_r0":1},
        "u_lgb_cat6":{"lgb9_r0":2,"cat6_r0":1},
        "u_lgb_cats":{"lgb9_r0":2,"cat2_r0":1,"cat6_r0":1},
        "u_cats":{"cat2_r0":1,"cat6_r0":1},
    },
    "KR":{
        "k_cat2r0":{"cat2_r0":1},"k_cat2r1":{"cat2_r1":1},
        "k_cat4r0":{"cat4_r0":1},"k_cat4r1":{"cat4_r1":1},
        "k_cat6r0":{"cat6_r0":1},"k_lgb3":{"lgb9_r3":1},
        "k_cat2":{"cat2_r0":1,"cat2_r1":1},
        "k_prior_cats":{"cat2_r0":1,"cat4_r0":1,"cat6_r0":1},
        "k_cat2_lgb":{"cat2_r1":2,"lgb9_r3":1},
        "k_all":{"cat2_r0":1,"cat2_r1":1,"cat4_r0":1,"cat6_r0":1,"lgb9_r3":1},
    },
}


def options(frame:pd.DataFrame,probability:np.ndarray,name:str)->list[dict]:
    y=frame.y.astype(int).to_numpy();up=float(y.mean());rows=[]
    for cutoff in np.arange(.30,.801,.005):
        pred=probability>=cutoff;tp=int((pred&(y==1)).sum());tn=int(((~pred)&(y==0)).sum())
        fp=int((pred&(y==0)).sum());fn=int(((~pred)&(y==1)).sum())
        bal=.5*(tp/(tp+fn)+tn/(tn+fp));acc=(tp+tn)/len(y)
        rows.append({"ensemble":name,"cutoff":float(cutoff),"n":len(y),
                     "tp":tp,"tn":tn,"fp":fp,"fn":fn,"balanced_accuracy":bal,
                     "accuracy":acc,"edge":acc-max(up,1-up),
                     "up_prediction_rate":float(pred.mean())})
    return rows


def shortlist(rows:list[dict])->list[dict]:
    selected=[]
    rankings=(
        sorted(rows,key=lambda r:8*r["balanced_accuracy"]+5*r["edge"],reverse=True)[:25],
        sorted([r for r in rows if r["balanced_accuracy"]>=.51],
               key=lambda r:r["accuracy"],reverse=True)[:20],
        sorted(rows,key=lambda r:r["balanced_accuracy"],reverse=True)[:15],
    )
    seen=set()
    for ranking in rankings:
        for row in ranking:
            key=(row["ensemble"],row["cutoff"])
            if key not in seen:selected.append(row);seen.add(key)
    return selected


def encoded(frame:pd.DataFrame,probability:np.ndarray,cutoff:float,market:str)->pd.DataFrame:
    z=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    z["prob"]=app.shift_probability(probability,cutoff)
    z["eval_weight"]=TARGET[market]/len(z)
    return z


def main()->None:
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V23_SEAL","NAVER_NEWS_V23_SEAL"}),"y"].notna().any():
        raise RuntimeError("V23 seal outcome exposure detected")
    raw=pd.read_parquet(ROOT/"cache"/"v23_augmented_oof.parquet")
    frames={
        "US":(data[data.source.eq("SEC_V23_DEV")&data.y.notna()].copy()
              .sort_values("event_time_utc").reset_index(drop=True)),
        "KR":(data[data.source.eq("NAVER_NEWS_V23_DEV")&data.y.notna()].copy()
              .sort_values("event_time_utc").reset_index(drop=True)),
    }
    probabilities={};market_options={}
    for market,frame in frames.items():
        wide=(raw[raw.market.eq(market)].pivot(index="event_id",columns="model_name",values="probability")
              .reindex(frame.event_id))
        probabilities[market]={}
        all_rows=[]
        for name,weights in ENSEMBLES[market].items():
            p=np.average(np.stack([wide[column].to_numpy(float) for column in weights]),
                         axis=0,weights=list(weights.values()))
            probabilities[market][name]=p;all_rows.extend(options(frame,p,name))
        market_options[market]=shortlist(all_rows)

    pairs=[]
    for us in market_options["US"]:
        for kr in market_options["KR"]:
            us_weight=TARGET["US"]/us["n"];kr_weight=TARGET["KR"]/kr["n"]
            tp=us["tp"]*us_weight+kr["tp"]*kr_weight
            tn=us["tn"]*us_weight+kr["tn"]*kr_weight
            fp=us["fp"]*us_weight+kr["fp"]*kr_weight
            fn=us["fn"]*us_weight+kr["fn"]*kr_weight
            total=sum(TARGET.values());bal=.5*(tp/(tp+fn)+tn/(tn+fp))
            accuracy=(tp+tn)/total;up_rate=(tp+fn)/total;edge=accuracy-max(up_rate,1-up_rate)
            required=(bal>=.54,edge>=.02,us["balanced_accuracy"]>=.51,
                      kr["balanced_accuracy"]>=.51)
            score=8*bal+5*edge
            pairs.append({"direction_pass_count":int(sum(required)),"direction_score":float(score),
                          "us":us,"kr":kr,"direction_balanced_accuracy":float(bal),
                          "direction_edge":float(edge)})
    pairs.sort(key=lambda r:(r["direction_pass_count"],r["direction_score"]),reverse=True)
    finalists=[]
    for row in pairs[:300]:
        combined=[]
        for market,choice in (("US",row["us"]),("KR",row["kr"])):
            combined.append(encoded(frames[market],probabilities[market][choice["ensemble"]],
                                    choice["cutoff"],market))
        metric=app.evaluate(pd.concat(combined,ignore_index=True),.80,.002)
        required=(metric["balanced_accuracy"]>=.54,metric["auc"]>=.56,
                  metric["edge_vs_naive"]>=.02,
                  metric["by_market"]["US"]["balanced_accuracy"]>=.51,
                  metric["by_market"]["KR"]["balanced_accuracy"]>=.51)
        row["pass_count"]=int(sum(required));row["score"]=(
            8*metric["balanced_accuracy"]+3*metric["auc"]+5*metric["edge_vs_naive"]
        );row["metrics"]=metric;finalists.append(row)
    finalists.sort(key=lambda r:(r["pass_count"],r["score"]),reverse=True)
    for row in finalists[:30]:
        combined=[]
        for market,choice in (("US",row["us"]),("KR",row["kr"])):
            combined.append(encoded(frames[market],probabilities[market][choice["ensemble"]],
                                    choice["cutoff"],market))
        row["bootstrap"]=app.bootstrap_by_day(pd.concat(combined,ignore_index=True),.80,.002,nboot=600)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),
            "ensembles":ENSEMBLES,"market_options":market_options,"pairs":finalists[:100]}
    path=ROOT/"cache"/"v23_direction_ensemble_audits.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for row in finalists[:20]:
        m=row["metrics"];print(
            f"passes={row['pass_count']}/5 US={row['us']['ensemble']}@{row['us']['cutoff']:.3f} "
            f"KR={row['kr']['ensemble']}@{row['kr']['cutoff']:.3f} "
            f"bal={m['balanced_accuracy']:.4f} auc={m['auc']:.4f} "
            f"edge={m['edge_vs_naive']:.4f} USbal={m['by_market']['US']['balanced_accuracy']:.4f} "
            f"KRbal={m['by_market']['KR']['balanced_accuracy']:.4f}",flush=True)
    print("RESULT="+str(path))


if __name__=="__main__":main()
