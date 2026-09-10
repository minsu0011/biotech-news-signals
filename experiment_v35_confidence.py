"""Leakage-safe V35 confidence and combined-policy search on opened outcomes."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app
import runtime_limits


ROOT=Path(__file__).resolve().parent
SEALED={"US_EXACT_V35_SEAL","KR_EXACT_V35_SEAL"}
PRIMARY={"US":{"US_EXACT_V34_SEAL"},"KR":{"KR_EXACT_V34_SEAL"}}
ANALOG={
    "US":{"US_EXACT_V31_DEV","US_EXACT_V31_SEAL"},
    "KR":{"KR_EXACT_V30_DEV","KR_EXACT_V30_SEAL",
          "KR_EXACT_V31_DEV","KR_EXACT_V31_SEAL"},
}
EXCLUDED={
    "year","month","weekday","hour","minute","minutes_from_open",
    "minutes_to_close","event_hour","event_weekday","event_month",
    "event_hour_sin","event_hour_cos","headline_len","body_len",
}
GATES=tuple(column for column in app.NUMERIC_COLS if column not in EXCLUDED)


def rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
    reference=np.asarray(reference,float);finite=np.isfinite(reference)
    fill=float(np.median(reference[finite])) if finite.any() else 0.0
    reference=np.sort(reference[finite])
    values=np.nan_to_num(np.asarray(values,float),nan=fill,posinf=fill,neginf=fill)
    if not len(reference):return np.full(len(values),0.5)
    left=np.searchsorted(reference,values,side="left")
    right=np.searchsorted(reference,values,side="right")
    return (left+right+1.0)/(2.0*len(reference))


def values(frame:pd.DataFrame,column:str)->np.ndarray:
    return pd.to_numeric(frame[column],errors="coerce").to_numpy(float)


def direction(reference:pd.DataFrame,target:pd.DataFrame,components:list)->np.ndarray:
    pieces=[]
    for column,sign,weight in components:
        piece=rank(values(reference,column),values(target,column))
        pieces.append(float(weight)*(piece if float(sign)>0 else 1.0-piece))
    return np.sum(pieces,axis=0)/sum(float(item[2]) for item in components)


def encode(reference:pd.DataFrame,target:pd.DataFrame,reference_direction:np.ndarray,
           target_direction:np.ndarray,cutoff:float,gate:dict)->np.ndarray:
    reference_margin=np.abs(reference_direction-cutoff)
    margin=rank(reference_margin,np.abs(target_direction-cutoff))
    reference_margin_rank=rank(reference_margin,reference_margin)
    reference_opportunity=float(gate["sign"])*values(reference,gate["column"])
    target_opportunity=float(gate["sign"])*values(target,gate["column"])
    opportunity=rank(reference_opportunity,target_opportunity)
    reference_opportunity_rank=rank(reference_opportunity,reference_opportunity)
    weight=float(gate["margin_weight"])
    reference_confidence=weight*reference_margin_rank+(1.0-weight)*reference_opportunity_rank
    confidence=weight*margin+(1.0-weight)*opportunity
    selected=rank(reference_confidence,confidence)>=float(gate["gate_quantile"])
    strength=np.where(selected,0.86+0.14*margin,0.84*margin)
    probability=np.where(target_direction>=cutoff,0.5+0.5*strength,0.5-0.5*strength)
    return np.clip(probability,1e-6,1.0-1e-6)


def high_conf(frame:pd.DataFrame,probability:np.ndarray)->dict|None:
    selected=np.maximum(probability,1.0-probability)>=0.925
    if selected.sum()<12:return None
    prediction=probability>=0.5;y=frame.y.astype(int).to_numpy()
    net=(np.where(prediction[selected],1.0,-1.0)*
         frame.fwd_ret_30m.to_numpy(float)[selected]-0.002)
    return {"n":int(selected.sum()),"coverage":float(selected.mean()),
            "accuracy":float((prediction[selected]==y[selected]).mean()),
            "net":float(net.mean())}


def prediction_frame(frame:pd.DataFrame,probability:np.ndarray,weight:float)->pd.DataFrame:
    output=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    output["prob"]=probability;output["eval_weight"]=float(weight)/len(output)
    return output


def main()->None:
    runtime_limits.configure();cfg=app.Config();data=app.load_search_frame(cfg)
    sealed=data.source.isin(SEALED)
    if data.loc[sealed,["y","fwd_ret_30m"]].notna().any().any():
        raise RuntimeError("V35 seal outcomes exposed")
    opened=data[data.y.notna()].copy()
    forbidden={"y","fwd_ret_30m","exit_price","exit_time_utc"}
    feature_only=pd.read_csv(app.LABELED_FILE,compression="gzip",dtype={"ticker":str},
        low_memory=False,usecols=lambda column:column not in forbidden)
    if forbidden&set(feature_only.columns):raise RuntimeError("forbidden V35 outcome parsed")
    feature_only=feature_only[feature_only.source.isin(SEALED)].copy()
    target_counts={market:int(feature_only.market.eq(market).sum()) for market in ("US","KR")}
    directions=json.loads((ROOT/"cache"/"v35_robust_direction.json").read_text(encoding="utf-8"))["rows"]
    local={};frames={};model_features={}
    for market in ("US","KR"):
        frames[market]={
            "primary":opened[opened.market.eq(market)&opened.source.isin(PRIMARY[market])]
                         .sort_values("event_time_utc").reset_index(drop=True),
            "analog":opened[opened.market.eq(market)&opened.source.isin(ANALOG[market])]
                        .sort_values("event_time_utc").reset_index(drop=True),
            "target":feature_only[feature_only.market.eq(market)]
                     .sort_values("event_time_utc").reset_index(drop=True),
        }
        model_features[market]={name:app.model_frame(frame) for name,frame in frames[market].items()}
        candidates=[];seen=set();ordered=[]
        ordered.extend(directions[market][:20])
        ordered.extend(sorted(directions[market],key=lambda row:(
            row["primary"]["balanced_accuracy"]+row["primary"]["auc"]+
            row["primary"]["edge"]),reverse=True)[:15])
        ordered.extend(sorted(directions[market],key=lambda row:row["primary"]["auc"],reverse=True)[:15])
        for row in ordered:
            key=json.dumps(row["components"],sort_keys=True)+f"|{row['cutoff']:.5f}"
            if key not in seen:candidates.append(row);seen.add(key)
            if len(candidates)==24:break
        policies=[]
        for direction_index,candidate in enumerate(candidates):
            raw={};reference_raw={}
            for name in ("primary","analog"):
                feature=model_features[market][name]
                reference_raw[name]=direction(feature,feature,candidate["components"])
                raw[name]=reference_raw[name]
            reference=model_features[market]["primary"];target=model_features[market]["target"]
            reference_raw["target"]=reference_raw["primary"]
            raw["target"]=direction(reference,target,candidate["components"])
            direction_policies=[]
            for column,sign,margin_weight,quantile in itertools.product(
                    GATES,(-1.0,1.0),(0.0,0.33,0.67,1.0),
                    (0.55,0.65,0.75,0.85)):
                gate={"column":column,"sign":sign,"margin_weight":margin_weight,
                      "gate_quantile":quantile}
                probabilities={name:encode(
                    model_features[market]["primary"] if name=="target" else model_features[market][name],
                    model_features[market][name],reference_raw[name],raw[name],
                    float(candidate["cutoff"]),gate) for name in ("primary","analog","target")}
                summaries={name:high_conf(frames[market][name],probabilities[name])
                           for name in ("primary","analog")}
                if any(value is None for value in summaries.values()):continue
                current=summaries["primary"];audit=summaries["analog"]
                target_coverage=float((np.maximum(probabilities["target"],
                                                   1.0-probabilities["target"])>=0.925).mean())
                checks=sum((current["accuracy"]>=0.60,current["net"]>0,
                            current["coverage"]>=0.15,target_coverage>=0.15,
                            audit["accuracy"]>=0.52,audit["net"]>-0.002))
                score=(3.0*current["accuracy"]+80.0*current["net"]+current["coverage"]+
                       0.6*audit["accuracy"]+10.0*audit["net"]+
                       candidate["primary"]["balanced_accuracy"]+
                       candidate["primary"]["auc"])
                direction_policies.append({
                    "market":market,"direction_index":direction_index,"direction":candidate,
                    "gate":gate,"summaries":summaries,"target_coverage":target_coverage,
                    "checks":int(checks),"score":float(score),"probabilities":probabilities,
                })
            direction_policies.sort(key=lambda item:(item["checks"],item["score"]),reverse=True)
            policies.extend(direction_policies[:12])
            if (direction_index+1)%10==0:
                print(f"GATES {market} {direction_index+1}/{len(candidates)}",flush=True)
        policies.sort(key=lambda item:(item["checks"],item["score"]),reverse=True)
        local[market]=policies[:60]
        for item in local[market][:12]:print("LOCAL",market,json.dumps(
            {key:value for key,value in item.items() if key!="probabilities"},ensure_ascii=False),flush=True)

    combined=[]
    for us,kr in itertools.product(local["US"],local["KR"]):
        current=pd.concat([
            prediction_frame(frames["US"]["primary"],us["probabilities"]["primary"],target_counts["US"]),
            prediction_frame(frames["KR"]["primary"],kr["probabilities"]["primary"],target_counts["KR"]),
        ],ignore_index=True)
        metrics=app.evaluate(current,0.925,cfg.round_trip_cost)
        feature_highconf=sum(
            int((np.maximum(item["probabilities"]["target"],
                            1.0-item["probabilities"]["target"])>=0.925).sum())
            for item in (us,kr))
        feature_coverage=float(feature_highconf/sum(target_counts.values()))
        checks=sum((metrics["balanced_accuracy"]>=cfg.cert_min_bal_acc,
                    metrics["auc"]>=cfg.cert_min_auc,
                    metrics["edge_vs_naive"]>=cfg.cert_min_edge_vs_naive,
                    metrics["highconf_coverage"]>=cfg.cert_min_highconf_coverage,
                    metrics["highconf_accuracy"]>=cfg.cert_min_highconf_acc,
                    metrics["strategy_mean_signed_net"]>0,
                    metrics["by_market"]["US"]["balanced_accuracy"]>=cfg.cert_min_market_bal_acc,
                    metrics["by_market"]["KR"]["balanced_accuracy"]>=cfg.cert_min_market_bal_acc,
                    feature_coverage>=cfg.cert_min_highconf_coverage))
        score=(metrics["balanced_accuracy"]+metrics["auc"]+metrics["edge_vs_naive"]+
               2.0*metrics["highconf_accuracy"]+80.0*metrics["strategy_mean_signed_net"]+
               0.2*min(metrics["by_market"][m]["balanced_accuracy"] for m in ("US","KR")))
        combined.append({"us":us,"kr":kr,"frame":current,"metrics":metrics,
                         "feature_only_highconf_n":feature_highconf,
                         "feature_only_highconf_coverage":feature_coverage,
                         "checks":int(checks),"score":float(score)})
    combined.sort(key=lambda item:(item["checks"],item["score"]),reverse=True)
    finalists=[]
    for item in combined[:15]:
        bootstrap=app.bootstrap_by_day(item["frame"],0.925,cfg.round_trip_cost,nboot=2000)
        checks=item["checks"]+int(bootstrap["balanced_accuracy_lower95"]>=cfg.cert_min_boot_bal_lower95)
        finalists.append({
            "policies":{
                "US":{key:value for key,value in item["us"].items() if key!="probabilities"},
                "KR":{key:value for key,value in item["kr"].items() if key!="probabilities"},
            },"sets":{"v34":{"metrics":item["metrics"],"bootstrap":bootstrap}},
            "feature_only_highconf_n":item["feature_only_highconf_n"],
            "feature_only_highconf_coverage":item["feature_only_highconf_coverage"],
            "checks":int(checks),"score":item["score"],
        })
    finalists.sort(key=lambda item:(item["checks"],item["score"]),reverse=True)
    for item in finalists[:20]:print("COMBINED",json.dumps(item,ensure_ascii=False),flush=True)
    path=ROOT/"cache"/"v35_confidence.json"
    path.write_text(json.dumps({"v35_seal_outcomes_loaded":False,
        "forbidden_outcome_columns_loaded":False,"target_counts":target_counts,
        "runtime":runtime_limits.status(),"policies":finalists},ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT="+str(path),flush=True)


if __name__=="__main__":main()
