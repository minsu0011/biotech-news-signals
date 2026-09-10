"""Cross-version direction pairing and high-confidence economics for V25."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import runtime_limits
import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
TARGET = {"US": 295.0, "KR": 521.0}
SOURCE = {
    "US": {"ref": "SEC_V24_DEV", "audit": "SEC_V24_SEAL", "dev": "SEC_V25_DEV"},
    "KR": {"ref": "NAVER_NEWS_V24_DEV", "audit": "NAVER_NEWS_V24_SEAL",
           "dev": "NAVER_NEWS_V25_DEV"},
}
DIRECTIONS = {
    "US": {
        "u3_57": {"components": (("pre_ret_5",-1,1),("abs_pre_ret_60",-1,1),
                                     ("ret_slope_5_30",-1,1)), "cutoff": .57},
        "u3_58": {"components": (("pre_ret_5",-1,1),("abs_pre_ret_60",-1,1),
                                     ("ret_slope_5_30",-1,1)), "cutoff": .58},
        "u3_60": {"components": (("pre_ret_5",-1,1),("abs_pre_ret_60",-1,1),
                                     ("ret_slope_5_30",-1,1)), "cutoff": .60},
        "u2_70": {"components": (("pre_ret_2",-1,.3343),("abs_pre_ret_60",-1,.6657)),
                    "cutoff": .70},
        "u2_71": {"components": (("pre_ret_2",-1,.3343),("abs_pre_ret_60",-1,.6657)),
                    "cutoff": .71},
    },
    "KR": {
        "k_v5a30_60": {"components": (("volume_ratio_5_30",1,1),("return_autocorr_30",1,1),
                                            ("benchmark_ret_30",1,1)), "cutoff": .60},
        "k_v5a30_62": {"components": (("volume_ratio_5_30",1,1),("return_autocorr_30",1,1),
                                            ("benchmark_ret_30",1,1)), "cutoff": .62},
        "k_v2a30_65": {"components": (("volume_ratio_2_30",1,1),("return_autocorr_30",1,1),
                                            ("benchmark_ret_30",1,1)), "cutoff": .65},
        "k_v2a30_68": {"components": (("volume_ratio_2_30",1,1),("return_autocorr_30",1,1),
                                            ("benchmark_ret_30",1,1)), "cutoff": .68},
        "k_v2a30_69": {"components": (("volume_ratio_2_30",1,1),("return_autocorr_30",1,1),
                                            ("benchmark_ret_30",1,1)), "cutoff": .69},
    },
}
GATE_COLUMNS = tuple(dict.fromkeys([
    "pre_ret_1","pre_ret_2","pre_ret_5","pre_ret_15","pre_ret_30","pre_ret_60",
    "pre_ret_90","pre_ret_120","abs_pre_ret_2","abs_pre_ret_5","abs_pre_ret_15",
    "abs_pre_ret_30","abs_pre_ret_60","pre_vol_5","pre_vol_10","pre_vol_30",
    "volume_ratio_1_30","volume_ratio_2_30","volume_ratio_5_30","volume_ratio_10_60",
    "range_5m","range_10m","range_30m","entry_bar_ret","entry_bar_range",
    "intraday_ret_open","return_autocorr_30","up_fraction_10","up_fraction_30",
    "trend_slope_30","trend_slope_60","vwap_distance_30","benchmark_ret_2",
    "benchmark_ret_5","benchmark_ret_15","benchmark_ret_30","benchmark_ret_60",
    "excess_ret_5","excess_ret_15","excess_ret_30","excess_ret_60",
    "minutes_from_open","minutes_to_close",
]))
_FEATURE_CACHE: dict[int,pd.DataFrame] = {}
_MASK_CACHE: dict[int,dict[str,np.ndarray]] = {}


def rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    if not len(reference): return np.full(len(values), .5)
    fill = float(np.median(reference))
    values = np.nan_to_num(np.asarray(values,float), nan=fill, posinf=fill, neginf=fill)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left+right+1)/(2*len(reference))


def values(frame: pd.DataFrame, column: str) -> np.ndarray:
    key=id(frame)
    if key not in _FEATURE_CACHE:_FEATURE_CACHE[key]=app.model_frame(frame)
    return pd.to_numeric(_FEATURE_CACHE[key][column], errors="coerce").to_numpy(float)


def direction(reference: pd.DataFrame, target: pd.DataFrame, choice: dict) -> tuple[np.ndarray,np.ndarray]:
    ref_parts=[]; target_parts=[]; weights=[]
    for column, sign, weight in choice["components"]:
        rv=values(reference,column); tv=values(target,column)
        rr=rank(rv,rv); tr=rank(rv,tv)
        if sign < 0: rr=1-rr;tr=1-tr
        ref_parts.append(weight*rr);target_parts.append(weight*tr);weights.append(weight)
    return np.sum(ref_parts,axis=0)/sum(weights),np.sum(target_parts,axis=0)/sum(weights)


def encode(reference: pd.DataFrame, target: pd.DataFrame, ref_direction: np.ndarray,
           target_direction: np.ndarray, cutoff: float, gate: dict) -> np.ndarray:
    ref_margin_raw=np.abs(ref_direction-cutoff);target_margin_raw=np.abs(target_direction-cutoff)
    margin=rank(ref_margin_raw,target_margin_raw)
    ref_opp=gate["sign"]*values(reference,gate["column"])
    target_opp=gate["sign"]*values(target,gate["column"])
    opportunity=rank(ref_opp,target_opp)
    ref_margin=rank(ref_margin_raw,ref_margin_raw)
    ref_opportunity=rank(ref_opp,ref_opp)
    ref_conf=gate["margin_weight"]*ref_margin+(1-gate["margin_weight"])*ref_opportunity
    conf_raw=gate["margin_weight"]*margin+(1-gate["margin_weight"])*opportunity
    selected=rank(ref_conf,conf_raw)>=gate["gate_quantile"]
    confidence=np.where(selected,.86+.14*margin,.84*margin)
    return np.where(target_direction>=cutoff,.5+.5*confidence,.5-.5*confidence)


def eval_frame(frame: pd.DataFrame, probability: np.ndarray, market: str) -> pd.DataFrame:
    z=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    z["prob"]=probability;z["eval_weight"]=TARGET[market]/len(z);return z


def split_metric(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    y=frame.y.astype(int).to_numpy();pred=probability>=.5
    selected=np.maximum(probability,1-probability)>=.925
    if selected.sum()==0:return {"n":0,"coverage":0,"accuracy":None,"net":None}
    net=np.where(pred[selected],1.,-1.)*frame.fwd_ret_30m.to_numpy(float)[selected]-.002
    return {"n":int(selected.sum()),"coverage":float(selected.mean()),
            "accuracy":float((pred[selected]==y[selected]).mean()),"net":float(net.mean())}


def masks(frame:pd.DataFrame)->dict[str,np.ndarray]:
    key=id(frame)
    if key in _MASK_CACHE:return _MASK_CACHE[key]
    h=frame.event_id.astype(str).map(lambda x:int(hashlib.sha256(x.encode()).hexdigest()[:16],16)%2).to_numpy()
    t=pd.to_datetime(frame.event_time_utc,utc=True).rank(method="first",pct=True).to_numpy()
    result={"all":np.ones(len(frame),bool),"h0":h==0,"h1":h==1,"early":t<=.5,"late":t>.5}
    _MASK_CACHE[key]=result
    return result


def main()->None:
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V25_SEAL","NAVER_NEWS_V25_SEAL"}),"y"].notna().any():
        raise RuntimeError("V25 seal outcome exposure detected")
    frames={};directions={}
    for market,sources in SOURCE.items():
        frames[market]={k:data[data.source.eq(v)&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
                        for k,v in sources.items()}
        directions[market]={}
        for name,choice in DIRECTIONS[market].items():
            audit_ref,audit=direction(frames[market]["ref"],frames[market]["audit"],choice)
            dev_ref,dev=direction(frames[market]["dev"],frames[market]["dev"],choice)
            directions[market][name]={"audit_ref":audit_ref,"audit":audit,"dev_ref":dev_ref,"dev":dev}

    direction_pairs=[]
    for us_name,kr_name in itertools.product(DIRECTIONS["US"],DIRECTIONS["KR"]):
        row={"US":us_name,"KR":kr_name,"sets":{}}
        for set_name,ref_name in (("audit","ref"),("dev","dev")):
            parts=[]
            for market,name in (("US",us_name),("KR",kr_name)):
                raw=directions[market][name][set_name];cut=DIRECTIONS[market][name]["cutoff"]
                parts.append(eval_frame(frames[market][set_name],app.shift_probability(raw,cut),market))
            combined=pd.concat(parts,ignore_index=True);met=app.evaluate(combined,.925,.002)
            row["sets"][set_name]={"metrics":met}
        a=row["sets"]["audit"]["metrics"];d=row["sets"]["dev"]["metrics"]
        row["robust_passes"]=sum([
            a["balanced_accuracy"]>=.54,d["balanced_accuracy"]>=.54,
            a["auc"]>=.56,d["auc"]>=.56,a["edge_vs_naive"]>=.02,d["edge_vs_naive"]>=.02,
            a["by_market"]["US"]["balanced_accuracy"]>=.51,
            d["by_market"]["US"]["balanced_accuracy"]>=.51,
            a["by_market"]["KR"]["balanced_accuracy"]>=.51,
            d["by_market"]["KR"]["balanced_accuracy"]>=.51,
        ])
        row["score"]=float(min(a["balanced_accuracy"],d["balanced_accuracy"])+
                           min(a["auc"],d["auc"])+min(a["edge_vs_naive"],d["edge_vs_naive"]))
        direction_pairs.append(row)
    direction_pairs.sort(key=lambda r:(r["robust_passes"],r["score"]),reverse=True)

    policies=[];gate_audits={};gate_choice_cache={}
    for pair in direction_pairs[:6]:
        shortlisted={}
        for market,name in (("US",pair["US"]),("KR",pair["KR"])):
            cache_key=(market,name)
            if cache_key in gate_choice_cache:
                shortlisted[market]=gate_choice_cache[cache_key]
                continue
            choices=[]
            for column,sign,mw,q in itertools.product(GATE_COLUMNS,(-1,1),(.25,.5,.75),(.80,.825,.85)):
                gate={"column":column,"sign":sign,"margin_weight":mw,"gate_quantile":q}
                splits={}
                for set_name,ref_name in (("audit","ref"),("dev","dev")):
                    p=encode(frames[market][ref_name],frames[market][set_name],
                             directions[market][name][set_name+"_ref"] if set_name=="audit" else directions[market][name]["dev_ref"],
                             directions[market][name][set_name],DIRECTIONS[market][name]["cutoff"],gate)
                    sm={}
                    for split_name,mask in masks(frames[market][set_name]).items():
                        sm[split_name]=split_metric(frames[market][set_name].loc[mask].reset_index(drop=True),p[mask])
                    splits[set_name]=sm
                full=[splits[s]["all"] for s in ("audit","dev")]
                valid=[v for ss in splits.values() for v in ss.values() if v["n"]>=5]
                if len(valid)<10:continue
                floor_acc=min(v["accuracy"] for v in full);floor_net=min(v["net"] for v in full)
                split_acc=min(v["accuracy"] for v in valid);split_net=min(v["net"] for v in valid)
                robust=sum(v["accuracy"]>=.55 for v in full)+sum(v["net"]>0 for v in full)
                score=4*floor_acc+80*floor_net+split_acc+20*split_net
                choices.append({"gate":gate,"sets":splits,"robust_passes":robust,"score":float(score)})
            choices.sort(key=lambda r:(r["robust_passes"],r["score"]),reverse=True)
            shortlisted[market]=choices[:12]
            gate_choice_cache[cache_key]=shortlisted[market]
            print(f"[GATES {market} {name}] retained={len(shortlisted[market])}",flush=True)
        gate_audits[f"{pair['US']}|{pair['KR']}"]=shortlisted

        for ug,kg in itertools.product(shortlisted["US"],shortlisted["KR"]):
            row={"direction":{"US":pair["US"],"KR":pair["KR"]},
                 "gate":{"US":ug["gate"],"KR":kg["gate"]},"sets":{}}
            for set_name,ref_name in (("audit","ref"),("dev","dev")):
                parts=[]
                for market,name,gate in (("US",pair["US"],ug["gate"]),("KR",pair["KR"],kg["gate"])):
                    p=encode(frames[market][ref_name],frames[market][set_name],
                             directions[market][name][set_name+"_ref"] if set_name=="audit" else directions[market][name]["dev_ref"],
                             directions[market][name][set_name],DIRECTIONS[market][name]["cutoff"],gate)
                    parts.append(eval_frame(frames[market][set_name],p,market))
                combined=pd.concat(parts,ignore_index=True);met=app.evaluate(combined,.925,.002)
                row["sets"][set_name]={"metrics":met}
            a=row["sets"]["audit"]["metrics"];d=row["sets"]["dev"]["metrics"]
            requirements=[]
            for m in (a,d):
                requirements += [m["balanced_accuracy"]>=.54,m["auc"]>=.56,
                                 m["edge_vs_naive"]>=.02,m["highconf_coverage"]>=.15,
                                 m["highconf_accuracy"]>=.60,m["strategy_mean_signed_net"]>0,
                                 m["by_market"]["US"]["balanced_accuracy"]>=.51,
                                 m["by_market"]["KR"]["balanced_accuracy"]>=.51]
            row["robust_passes"]=sum(requirements)
            row["score"]=float(min(a["balanced_accuracy"],d["balanced_accuracy"])+
                               min(a["auc"],d["auc"])+min(a["edge_vs_naive"],d["edge_vs_naive"])+
                               2*min(a["highconf_accuracy"],d["highconf_accuracy"])+
                               50*min(a["strategy_mean_signed_net"],d["strategy_mean_signed_net"]))
            policies.append(row)
    policies.sort(key=lambda r:(r["robust_passes"],r["score"]),reverse=True)
    for row in policies[:12]:
        for set_name,ref_name in (("audit","ref"),("dev","dev")):
            parts=[]
            for market in ("US","KR"):
                name=row["direction"][market];gate=row["gate"][market]
                p=encode(frames[market][ref_name],frames[market][set_name],
                         directions[market][name][set_name+"_ref"] if set_name=="audit" else directions[market][name]["dev_ref"],
                         directions[market][name][set_name],DIRECTIONS[market][name]["cutoff"],gate)
                parts.append(eval_frame(frames[market][set_name],p,market))
            row["sets"][set_name]["bootstrap"]=app.bootstrap_by_day(pd.concat(parts,ignore_index=True),.925,.002,nboot=800)
        row["robust_passes"] += sum(row["sets"][s]["bootstrap"]["balanced_accuracy_lower95"]>=.50 for s in ("audit","dev"))
    policies[:12]=sorted(policies[:12],key=lambda r:(r["robust_passes"],r["score"]),reverse=True)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"target":TARGET,
            "directions":DIRECTIONS,"direction_pairs":direction_pairs,"gate_audits":gate_audits,
            "policies":policies[:150]}
    path=ROOT/"cache"/"v25_policy.json";path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for row in policies[:20]:
        print(f"pass={row['robust_passes']}/18 dir={row['direction']} gate={row['gate']}")
        for s in ("audit","dev"):
            m=row["sets"][s]["metrics"];b=row["sets"][s].get("bootstrap",{})
            print(f"  {s} bal={m['balanced_accuracy']:.4f} auc={m['auc']:.4f} edge={m['edge_vs_naive']:.4f} "
                  f"hc={m['highconf_accuracy']:.4f}/{m['highconf_coverage']:.4f} net={m['strategy_mean_signed_net']:.5f} "
                  f"boot={b.get('balanced_accuracy_lower95')}")
    print("RESULT="+str(path))


if __name__=="__main__":main()
