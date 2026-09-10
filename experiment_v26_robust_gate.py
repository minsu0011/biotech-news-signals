"""Joint current/rolling confidence search for deterministic V26 policy."""
from __future__ import annotations
import itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent;TARGET={"US":909.,"KR":450.}
US_COMPONENTS=(("pre_ret_10",-1,.1668246172331678),("close_position_10",-1,.26730617901774384),
 ("benchmark_ret_15",1,.3370659178609464),("abs_pre_ret_30",-1,.09691528546067942),
 ("abs_pre_ret_60",-1,.1318880004274626))
GATES=("pre_ret_1","pre_ret_2","pre_ret_3","pre_ret_5","pre_ret_10","pre_ret_15","pre_ret_30","pre_ret_60",
 "pre_ret_90","pre_ret_120","abs_pre_ret_2","abs_pre_ret_5","abs_pre_ret_15","abs_pre_ret_30","abs_pre_ret_60",
 "pre_vol_5","pre_vol_10","pre_vol_30","volume_ratio_1_30","volume_ratio_2_30","volume_ratio_5_30",
 "volume_ratio_10_60","range_5m","range_10m","range_30m","entry_bar_ret","entry_bar_range","intraday_ret_open",
 "return_autocorr_30","up_fraction_10","up_fraction_30","trend_slope_30","trend_slope_60","vwap_distance_30",
 "benchmark_ret_2","benchmark_ret_5","benchmark_ret_15","benchmark_ret_30","benchmark_ret_60","excess_ret_5",
 "excess_ret_15","excess_ret_30","excess_ret_60","minutes_from_open","minutes_to_close")

def rank(ref,v):
 ref=np.sort(np.asarray(ref,float)[np.isfinite(ref)]);fill=float(np.median(ref)) if len(ref) else 0
 v=np.nan_to_num(np.asarray(v,float),nan=fill,posinf=fill,neginf=fill)
 if not len(ref):return np.full(len(v),.5)
 l=np.searchsorted(ref,v,"left");r=np.searchsorted(ref,v,"right");return (l+r+1)/(2*len(ref))
def val(f,c):return pd.to_numeric(f[c],errors="coerce").to_numpy(float)
def score(rf,tf):
 out=[]
 for c,s,w in US_COMPONENTS:
  rr=val(rf,c);z=rank(rr,val(tf,c));out.append(w*(z if s>0 else 1-z))
 return np.sum(out,axis=0)/sum(c[2] for c in US_COMPONENTS)
def encode(rf,tf,rr,tr,cut,g):
 rm=np.abs(rr-cut);margin=rank(rm,np.abs(tr-cut));ro=g["sign"]*val(rf,g["column"]);opp=rank(ro,g["sign"]*val(tf,g["column"]))
 rmr=rank(rm,rm);ror=rank(ro,ro);rc=g["margin_weight"]*rmr+(1-g["margin_weight"])*ror
 conf=g["margin_weight"]*margin+(1-g["margin_weight"])*opp;selected=rank(rc,conf)>=g["gate_quantile"]
 strength=np.where(selected,.86+.14*margin,.84*margin);return np.where(tr>=cut,.5+.5*strength,.5-.5*strength)
def part(frame,p,m):
 z=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy();z["prob"]=p;z["eval_weight"]=TARGET[m]/len(z);return z
def hc(frame,p):
 s=np.maximum(p,1-p)>=.925;pred=p>=.5;y=frame.y.astype(int).to_numpy()
 if s.sum()<8:return None
 net=np.where(pred[s],1.,-1.)*frame.fwd_ret_30m.to_numpy(float)[s]-.002
 return {"n":int(s.sum()),"coverage":float(s.mean()),"accuracy":float((pred[s]==y[s]).mean()),"net":float(net.mean())}

def main():
 data=app.load_search_frame(app.Config())
 if data.loc[data.source.isin({"SEC_V26_SEAL","NAVER_NEWS_V26_SEAL"}),"y"].notna().any():raise RuntimeError("V26 seal exposed")
 frames={
  "current":{"US":data[data.source.eq("SEC_V26_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),
             "KR":data[data.source.eq("NAVER_NEWS_V26_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)},
  "audit":{"US":data[data.source.eq("SEC_V24_SEAL")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),
           "KR":data[data.source.isin({"NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL"})&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)}}
 refs={"current":{"US":frames["current"]["US"],"KR":frames["current"]["KR"]},
       "audit":{"US":data[data.source.eq("SEC_V24_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True),"KR":frames["audit"]["KR"]}}
 ff={s:{m:app.model_frame(f) for m,f in by.items()} for s,by in frames.items()};rf={s:{m:app.model_frame(f) for m,f in by.items()} for s,by in refs.items()}
 raw={"current":{},"audit":{}};rawref={"current":{},"audit":{}}
 for s in ("current","audit"):
  rawref[s]["US"]=score(rf[s]["US"],rf[s]["US"]);raw[s]["US"]=score(rf[s]["US"],ff[s]["US"])
 current=pd.read_parquet(ROOT/"cache"/"v26_model_oof.parquet");cw=current[current.market.eq("KR")].pivot(index="event_id",columns="model_name",values="probability")
 rolling=pd.read_parquet(ROOT/"cache"/"v26_rolling_kr_oof.parquet");rw=rolling.pivot(index="event_id",columns="model_name",values="probability")
 raw["current"]["KR"]=cw["v24v25_lgb15_r3"].reindex(frames["current"]["KR"].event_id).to_numpy(float)
 raw["audit"]["KR"]=rw["lgb15_r3"].reindex(frames["audit"]["KR"].event_id).to_numpy(float)
 rawref["current"]["KR"]=raw["current"]["KR"];rawref["audit"]["KR"]=raw["audit"]["KR"]
 cuts={"US":.54,"KR":.63};short={}
 for m in ("US","KR"):
  rows=[]
  for col,sign,mw,q in itertools.product(GATES,(-1,1),(.25,.5,.75),(.80,.825,.85)):
   g={"column":col,"sign":sign,"margin_weight":mw,"gate_quantile":q};sets={}
   for s in ("current","audit"):
    p=encode(rf[s][m],ff[s][m],rawref[s][m],raw[s][m],cuts[m],g);sets[s]=hc(frames[s][m],p)
   if any(v is None for v in sets.values()):continue
   robust=sum([sets["current"]["accuracy"]>=.6,sets["audit"]["accuracy"]>=.57,
               sets["current"]["net"]>0,sets["audit"]["net"]>0])
   sc=4*min(v["accuracy"] for v in sets.values())+100*min(v["net"] for v in sets.values())+min(v["coverage"] for v in sets.values())
   rows.append({"gate":g,"sets":sets,"robust":robust,"score":float(sc)})
  rows.sort(key=lambda r:(r["robust"],r["score"]),reverse=True);short[m]=rows[:60]
  print("SHORT",m,len(rows),json.dumps(rows[:5],ensure_ascii=False),flush=True)
 policies=[]
 for ug,kg in itertools.product(short["US"],short["KR"]):
  row={"gates":{"US":ug,"KR":kg},"sets":{}}
  for s in ("current","audit"):
   parts=[]
   for m,g in (("US",ug["gate"]),("KR",kg["gate"])):
    p=encode(rf[s][m],ff[s][m],rawref[s][m],raw[s][m],cuts[m],g);parts.append(part(frames[s][m],p,m))
   c=pd.concat(parts,ignore_index=True);met=app.evaluate(c,.925,.002);row["sets"][s]={"metrics":met}
  req=[]
  for s in ("current","audit"):
   x=row["sets"][s]["metrics"];req += [x["balanced_accuracy"]>=.54,x["auc"]>=.56,x["edge_vs_naive"]>=.02,
     x["highconf_coverage"]>=.15,x["highconf_accuracy"]>=.60,x["strategy_mean_signed_net"]>0,
     x["by_market"]["US"]["balanced_accuracy"]>=.51,x["by_market"]["KR"]["balanced_accuracy"]>=.51]
  row["passes"]=sum(req);a=row["sets"]["audit"]["metrics"];c=row["sets"]["current"]["metrics"]
  row["score"]=float(min(a["balanced_accuracy"],c["balanced_accuracy"])+min(a["auc"],c["auc"])+min(a["edge_vs_naive"],c["edge_vs_naive"])+
                     2*min(a["highconf_accuracy"],c["highconf_accuracy"])+50*min(a["strategy_mean_signed_net"],c["strategy_mean_signed_net"]))
  policies.append(row)
 policies.sort(key=lambda r:(r["passes"],r["score"]),reverse=True)
 for row in policies[:20]:
  for s in ("current","audit"):
   parts=[]
   for m in ("US","KR"):
    g=row["gates"][m]["gate"];p=encode(rf[s][m],ff[s][m],rawref[s][m],raw[s][m],cuts[m],g);parts.append(part(frames[s][m],p,m))
   row["sets"][s]["bootstrap"]=app.bootstrap_by_day(pd.concat(parts),.925,.002,nboot=1200)
  row["passes"]+=sum(row["sets"][s]["bootstrap"]["balanced_accuracy_lower95"]>=.50 for s in ("current","audit"))
 policies[:20]=sorted(policies[:20],key=lambda r:(r["passes"],r["score"]),reverse=True)
 result={"v26_seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"direction":{"US":{"components":US_COMPONENTS,"cutoff":.54},"KR":{"model":"rolling lgb15 r3","cutoff":.63}},"short":short,"policies":policies[:150]}
 path=ROOT/"cache"/"v26_robust_gate.json";path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 for r in policies[:20]:print(json.dumps(r,ensure_ascii=False))
 print("RESULT="+str(path))
if __name__=="__main__":main()
