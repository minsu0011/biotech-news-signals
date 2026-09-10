"""V24 fixed OOF probability ensembles and market-cutoff pairing."""
from __future__ import annotations

import runtime_limits

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


ROOT=Path(__file__).resolve().parent
TARGET={"US":438.0,"KR":485.0}
ENSEMBLES={
    "US":{
        "u_cat2":{"v23_cat2_r3":1},
        "u_lgb5":{"v23_lgb5_r3":1},
        "u_lgb9":{"v23_lgb9_r3":1},
        "u_lgb15":{"v23_lgb15_r3":1},
        "u_lgbs":{"v23_lgb5_r3":1,"v23_lgb9_r3":1,"v23_lgb15_r3":1},
        "u_cat_lgb5":{"v23_cat2_r3":1,"v23_lgb5_r3":1},
        "u_cat_lgb9":{"v23_cat2_r3":1,"v23_lgb9_r3":1},
        "u_all":{"v23_cat2_r3":2,"v23_lgb5_r3":1,
                 "v23_lgb9_r3":1,"v23_lgb15_r3":1},
    },
    "KR":{
        "k_cat6":{"v22v23_cat6_r0":1},
        "k_cat2":{"v22v23_cat2_r0":1},
        "k_cat6r1":{"v22v23_cat6_r1":1},
        "k_lgb5":{"v22v23_lgb5_r0":1},
        "k_lgb15":{"v22v23_lgb15_r0":1},
        "k_lgbs":{"v22v23_lgb5_r0":1,"v22v23_lgb15_r0":1},
        "k_cat_lgb":{"v22v23_cat6_r0":1,"v22v23_lgb15_r0":1},
        "k_all":{"v22v23_cat6_r0":2,"v22v23_cat2_r0":1,
                 "v22v23_lgb5_r0":1,"v22v23_lgb15_r0":1},
    },
}


def ensemble(wide:pd.DataFrame,weights:dict[str,int])->np.ndarray:
    return np.average(np.stack([wide[name].to_numpy(float) for name in weights]),
                      axis=0,weights=list(weights.values()))


def eval_part(frame:pd.DataFrame,raw:np.ndarray,cutoff:float,market:str)->tuple[pd.DataFrame,dict]:
    z=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    z["prob"]=app.shift_probability(raw,cutoff);z["eval_weight"]=TARGET[market]/len(z)
    return z,app.evaluate(z,.80,.002)


def main()->None:
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V24_SEAL","NAVER_NEWS_V24_SEAL"}),"y"].notna().any():
        raise RuntimeError("V24 seal outcome exposure detected")
    raw_cache=pd.read_parquet(ROOT/"cache"/"v24_model_oof.parquet")
    frames={
        "US":data[data.source.eq("SEC_V24_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),
        "KR":data[data.source.eq("NAVER_NEWS_V24_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),
    }
    raw={};options={}
    for market,frame in frames.items():
        wide=(raw_cache[raw_cache.market.eq(market)]
              .pivot(index="event_id",columns="model_name",values="probability")
              .reindex(frame.event_id))
        raw[market]={name:ensemble(wide,weights) for name,weights in ENSEMBLES[market].items()}
        rows=[]
        for name,probability in raw[market].items():
            for cutoff in np.arange(.25,.751,.005):
                _,metric=eval_part(frame,probability,float(cutoff),market)
                bm=metric["by_market"][market]
                score=(8*bm["balanced_accuracy"]+5*metric["edge_vs_naive"]+
                       2*(metric["auc"] if np.isfinite(metric["auc"]) else .5))
                rows.append({"ensemble":name,"cutoff":float(cutoff),"score":float(score),
                             "balanced_accuracy":bm["balanced_accuracy"],
                             "accuracy":metric["accuracy"],"auc":metric["auc"],
                             "edge":metric["edge_vs_naive"]})
        selected=[];seen=set()
        rankings=(
            sorted(rows,key=lambda r:r["score"],reverse=True)[:50],
            sorted([r for r in rows if r["balanced_accuracy"]>=.54],key=lambda r:r["edge"],reverse=True)[:40],
            sorted(rows,key=lambda r:r["balanced_accuracy"],reverse=True)[:30],
        )
        for ranking in rankings:
            for row in ranking:
                key=(row["ensemble"],row["cutoff"])
                if key not in seen:selected.append(row);seen.add(key)
        options[market]=selected

    pairs=[]
    for us in options["US"]:
        us_frame,_=eval_part(frames["US"],raw["US"][us["ensemble"]],us["cutoff"],"US")
        for kr in options["KR"]:
            kr_frame,_=eval_part(frames["KR"],raw["KR"][kr["ensemble"]],kr["cutoff"],"KR")
            combined=pd.concat([us_frame,kr_frame],ignore_index=True)
            metric=app.evaluate(combined,.80,.002)
            required=(metric["balanced_accuracy"]>=.54,metric["auc"]>=.56,
                      metric["edge_vs_naive"]>=.02,
                      metric["by_market"]["US"]["balanced_accuracy"]>=.51,
                      metric["by_market"]["KR"]["balanced_accuracy"]>=.51)
            score=(8*metric["balanced_accuracy"]+3*metric["auc"]+
                   5*metric["edge_vs_naive"])
            pairs.append({"pass_count":int(sum(required)),"score":float(score),
                          "us":us,"kr":kr,"metrics":metric})
    pairs.sort(key=lambda row:(row["pass_count"],row["score"]),reverse=True)
    for row in pairs[:40]:
        parts=[]
        for market,choice in (("US",row["us"]),("KR",row["kr"])):
            part,_=eval_part(frames[market],raw[market][choice["ensemble"]],choice["cutoff"],market)
            parts.append(part)
        row["bootstrap"]=app.bootstrap_by_day(pd.concat(parts,ignore_index=True),.80,.002,nboot=600)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"target":TARGET,
            "ensembles":ENSEMBLES,"options":options,"pairs":pairs[:100]}
    path=ROOT/"cache"/"v24_direction_audits.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for row in pairs[:20]:
        m=row["metrics"]
        print(f"passes={row['pass_count']}/5 US={row['us']['ensemble']}@{row['us']['cutoff']:.3f} "
              f"KR={row['kr']['ensemble']}@{row['kr']['cutoff']:.3f} "
              f"bal={m['balanced_accuracy']:.4f} auc={m['auc']:.4f} edge={m['edge_vs_naive']:.4f} "
              f"US={m['by_market']['US']['balanced_accuracy']:.4f} "
              f"KR={m['by_market']['KR']['balanced_accuracy']:.4f} "
              f"boot={row.get('bootstrap',{}).get('balanced_accuracy_lower95')}",flush=True)
    print("RESULT="+str(path))


if __name__=="__main__":main()
