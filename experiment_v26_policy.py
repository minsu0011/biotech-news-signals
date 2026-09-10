"""V26 market pairing and entry-time confidence gate search."""
from __future__ import annotations
import hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent;TARGET={"US":909.0,"KR":450.0}
GATE_COLS=("pre_ret_1","pre_ret_2","pre_ret_3","pre_ret_5","pre_ret_10","pre_ret_15","pre_ret_30",
 "pre_ret_60","pre_ret_90","pre_ret_120","abs_pre_ret_2","abs_pre_ret_5","abs_pre_ret_15",
 "abs_pre_ret_30","abs_pre_ret_60","pre_vol_5","pre_vol_10","pre_vol_30","volume_ratio_1_30",
 "volume_ratio_2_30","volume_ratio_5_30","volume_ratio_10_60","range_5m","range_10m","range_30m",
 "entry_bar_ret","entry_bar_range","intraday_ret_open","return_autocorr_30","up_fraction_10",
 "up_fraction_30","trend_slope_30","trend_slope_60","vwap_distance_30","benchmark_ret_2",
 "benchmark_ret_5","benchmark_ret_15","benchmark_ret_30","benchmark_ret_60","excess_ret_5",
 "excess_ret_15","excess_ret_30","excess_ret_60","minutes_from_open","minutes_to_close")

def rank(ref,v):
    ref=np.sort(np.asarray(ref,float)[np.isfinite(ref)]);fill=float(np.median(ref)) if len(ref) else 0
    v=np.nan_to_num(np.asarray(v,float),nan=fill,posinf=fill,neginf=fill)
    if not len(ref):return np.full(len(v),.5)
    l=np.searchsorted(ref,v,"left");r=np.searchsorted(ref,v,"right");return (l+r+1)/(2*len(ref))

def vals(features,c):return pd.to_numeric(features[c],errors="coerce").to_numpy(float)

def rank_score(features,components):
    pieces=[];weights=[]
    for c,s,w in components:
        v=vals(features,c);z=rank(v,v);pieces.append(w*(z if s>0 else 1-z));weights.append(w)
    return np.sum(pieces,axis=0)/sum(weights)

def eval_frame(frame,p,market):
    z=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    z["prob"]=p;z["eval_weight"]=TARGET[market]/len(z);return z

def masks(frame):
    h=frame.event_id.astype(str).map(lambda x:int(hashlib.sha256(x.encode()).hexdigest()[:16],16)%2).to_numpy()
    t=pd.to_datetime(frame.event_time_utc,utc=True).rank(method="first",pct=True).to_numpy()
    return {"all":np.ones(len(frame),bool),"h0":h==0,"h1":h==1,"early":t<=.5,"late":t>.5}

def basic(frame,p):
    y=frame.y.astype(int).to_numpy();pred=p>=.5;pos=y.sum();neg=len(y)-pos
    tp=((pred==1)&(y==1)).sum();tn=((pred==0)&(y==0)).sum();acc=(tp+tn)/len(y)
    return {"n":len(y),"accuracy":float(acc),"balanced_accuracy":float(.5*(tp/pos+tn/neg)),
            "edge":float(acc-max(y.mean(),1-y.mean()))}

def encode(features,raw,cut,gate):
    margin_raw=np.abs(raw-cut);margin=rank(margin_raw,margin_raw)
    opp_raw=gate["sign"]*vals(features,gate["column"]);opp=rank(opp_raw,opp_raw)
    conf_raw=gate["margin_weight"]*margin+(1-gate["margin_weight"])*opp
    selected=rank(conf_raw,conf_raw)>=gate["gate_quantile"]
    confidence=np.where(selected,.86+.14*margin,.84*margin)
    return np.where(raw>=cut,.5+.5*confidence,.5-.5*confidence)

def gate_metric(frame,p,mask):
    selected=(np.maximum(p,1-p)>=.925)&mask;pred=p>=.5;y=frame.y.astype(int).to_numpy()
    if selected.sum()<5:return None
    net=np.where(pred[selected],1.,-1.)*frame.fwd_ret_30m.to_numpy(float)[selected]-.002
    return {"n":int(selected.sum()),"coverage":float(selected.sum()/mask.sum()),
            "accuracy":float((pred[selected]==y[selected]).mean()),"net":float(net.mean())}

def main():
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.isin({"SEC_V26_SEAL","NAVER_NEWS_V26_SEAL"}),"y"].notna().any():raise RuntimeError("V26 seal exposed")
    frames={"US":data[data.source.eq("SEC_V26_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),
            "KR":data[data.source.eq("NAVER_NEWS_V26_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)}
    features={m:app.model_frame(f) for m,f in frames.items()};split_masks={m:masks(f) for m,f in frames.items()}
    rank_json=json.loads((ROOT/"cache"/"v26_rank.json").read_text(encoding="utf-8"))
    us_choices=[{"name":"u_equal3","components":[["close_position_10",-1,1],["benchmark_ret_15",1,1],["abs_pre_ret_60",-1,1]],"cutoff":.55}]
    seen=set()
    for i,row in enumerate(rank_json["candidates"][:80]):
        key=(tuple((c["column"],c["sign"],round(c["weight"],5)) for c in row["components"]),round(row["cutoff"],3))
        if key in seen:continue
        seen.add(key);us_choices.append({"name":f"u_rank{i}","components":[[c["column"],c["sign"],c["weight"]] for c in row["components"]],"cutoff":row["cutoff"]})
        if len(us_choices)>=21:break
    us_raw={c["name"]:rank_score(features["US"],c["components"]) for c in us_choices}
    us_meta={c["name"]:c for c in us_choices}
    cache=pd.read_parquet(ROOT/"cache"/"v26_model_oof.parquet")
    kwide=(cache[cache.market.eq("KR")].pivot(index="event_id",columns="model_name",values="probability").reindex(frames["KR"].event_id))
    audits=json.loads((ROOT/"cache"/"v26_model_audits.json").read_text(encoding="utf-8"))
    allowed_prefix=("v24v25_cat2_","v24v25_cat4_r1","v24v25_cat6_r1","v24v25_lgb3_r1",
                    "v24v25_lgb5_r1","v24v25_lgb9_","v24v25_lgb15_")
    kr_choices=[]
    for model in audits["markets"]["KR"]["models"]:
        if not model["name"].startswith(allowed_prefix):continue
        for choice in model["choices"][:12]:
            kr_choices.append({"name":model["name"],"cutoff":choice["cutoff"]})
    directions=[]
    for uc,kc in itertools.product(us_choices,kr_choices):
        upr=app.shift_probability(us_raw[uc["name"]],uc["cutoff"])
        kraw=kwide[kc["name"]].to_numpy(float);kpr=app.shift_probability(kraw,kc["cutoff"])
        combined=pd.concat([eval_frame(frames["US"],upr,"US"),eval_frame(frames["KR"],kpr,"KR")],ignore_index=True)
        met=app.evaluate(combined,.925,.002);splits={}
        for market,p in (("US",upr),("KR",kpr)):
            splits[market]={name:basic(frames[market].loc[mask].reset_index(drop=True),p[mask])
                            for name,mask in split_masks[market].items()}
        passes=sum([met["balanced_accuracy"]>=.54,met["auc"]>=.56,met["edge_vs_naive"]>=.02,
                    met["by_market"]["US"]["balanced_accuracy"]>=.51,met["by_market"]["KR"]["balanced_accuracy"]>=.51])
        passes+=sum(v["balanced_accuracy"]>=.50 for m in splits.values() for n,v in m.items() if n!="all")
        score=8*met["balanced_accuracy"]+3*met["auc"]+5*met["edge_vs_naive"]+min(v["balanced_accuracy"] for m in splits.values() for v in m.values())
        directions.append({"US":uc,"KR":kc,"metrics":met,"split":splits,"passes":passes,"score":float(score)})
    directions.sort(key=lambda r:(r["passes"],r["score"]),reverse=True)
    for row in directions[:30]:
        upr=app.shift_probability(us_raw[row["US"]["name"]],row["US"]["cutoff"])
        kpr=app.shift_probability(kwide[row["KR"]["name"]].to_numpy(float),row["KR"]["cutoff"])
        row["bootstrap"]=app.bootstrap_by_day(pd.concat([eval_frame(frames["US"],upr,"US"),eval_frame(frames["KR"],kpr,"KR")]),.925,.002,nboot=800)

    policies=[];gate_cache={}
    for direction_row in directions[:8]:
        shortlisted={}
        for market in ("US","KR"):
            raw=us_raw[direction_row["US"]["name"]] if market=="US" else kwide[direction_row["KR"]["name"]].to_numpy(float)
            cut=direction_row[market]["cutoff"];key=(market,direction_row[market]["name"],cut)
            if key in gate_cache:shortlisted[market]=gate_cache[key];continue
            rows=[]
            for col,sign,mw,q in itertools.product(GATE_COLS,(-1,1),(.25,.5,.75),(.80,.825,.85)):
                gate={"column":col,"sign":sign,"margin_weight":mw,"gate_quantile":q};p=encode(features[market],raw,cut,gate)
                split={name:gate_metric(frames[market],p,mask) for name,mask in split_masks[market].items()}
                if any(v is None for v in split.values()):continue
                full=split["all"];floor_acc=min(v["accuracy"] for v in split.values());floor_net=min(v["net"] for v in split.values())
                passes=sum([full["accuracy"]>=.6,full["net"]>0,floor_acc>=.5,floor_net>-.002])
                score=4*full["accuracy"]+100*full["net"]+floor_acc+20*floor_net
                rows.append({"gate":gate,"split":split,"passes":passes,"score":float(score)})
            rows.sort(key=lambda r:(r["passes"],r["score"]),reverse=True);gate_cache[key]=rows[:18];shortlisted[market]=rows[:18]
        for ug,kg in itertools.product(shortlisted["US"],shortlisted["KR"]):
            up=encode(features["US"],us_raw[direction_row["US"]["name"]],direction_row["US"]["cutoff"],ug["gate"])
            kp=encode(features["KR"],kwide[direction_row["KR"]["name"]].to_numpy(float),direction_row["KR"]["cutoff"],kg["gate"])
            combined=pd.concat([eval_frame(frames["US"],up,"US"),eval_frame(frames["KR"],kp,"KR")],ignore_index=True)
            met=app.evaluate(combined,.925,.002);req=[met["balanced_accuracy"]>=.54,met["auc"]>=.56,
                met["edge_vs_naive"]>=.02,met["highconf_coverage"]>=.15,met["highconf_accuracy"]>=.60,
                met["strategy_mean_signed_net"]>0,met["by_market"]["US"]["balanced_accuracy"]>=.51,
                met["by_market"]["KR"]["balanced_accuracy"]>=.51]
            score=4*met["balanced_accuracy"]+2*met["auc"]+3*met["highconf_accuracy"]+50*met["strategy_mean_signed_net"]+met["edge_vs_naive"]
            policies.append({"direction":{"US":direction_row["US"],"KR":direction_row["KR"]},"gates":{"US":ug,"KR":kg},
                             "metrics":met,"passes":sum(req),"score":float(score)})
    policies.sort(key=lambda r:(r["passes"],r["score"]),reverse=True)
    for row in policies[:20]:
        dr=row["direction"];up=encode(features["US"],us_raw[dr["US"]["name"]],dr["US"]["cutoff"],row["gates"]["US"]["gate"])
        kp=encode(features["KR"],kwide[dr["KR"]["name"]].to_numpy(float),dr["KR"]["cutoff"],row["gates"]["KR"]["gate"])
        row["bootstrap"]=app.bootstrap_by_day(pd.concat([eval_frame(frames["US"],up,"US"),eval_frame(frames["KR"],kp,"KR")]),.925,.002,nboot=1200)
        row["passes"]+=int(row["bootstrap"]["balanced_accuracy_lower95"]>=.50)
    policies[:20]=sorted(policies[:20],key=lambda r:(r["passes"],r["score"]),reverse=True)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"directions":directions[:120],"policies":policies[:150]}
    path=ROOT/"cache"/"v26_policy.json";path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("DIRECTIONS")
    for r in directions[:15]:print(r["passes"],r["US"]["name"],r["US"]["cutoff"],r["KR"],r["metrics"]["balanced_accuracy"],r["metrics"]["auc"],r["metrics"]["edge_vs_naive"],r.get("bootstrap"))
    print("POLICIES")
    for r in policies[:20]:print(json.dumps(r,ensure_ascii=False))
    print("RESULT="+str(path))

if __name__=="__main__":main()
