"""Low-dimensional V26 US rank search with V24 and internal split audits."""
from __future__ import annotations
import hashlib,itertools,json,heapq
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent;FEATURES=tuple(app.NUMERIC_COLS)

def rank(ref,v):
    ref=np.sort(ref[np.isfinite(ref)]);fill=float(np.median(ref)) if len(ref) else 0
    v=np.nan_to_num(v,nan=fill,posinf=fill,neginf=fill)
    if not len(ref):return np.full(len(v),.5)
    l=np.searchsorted(ref,v,"left");r=np.searchsorted(ref,v,"right")
    return (l+r+1)/(2*len(ref))

def matrix(ref,target):
    a=app.model_frame(ref);b=app.model_frame(target);cols=[]
    for c in FEATURES:
        rv=pd.to_numeric(a[c],errors="coerce").to_numpy(float)
        tv=pd.to_numeric(b[c],errors="coerce").to_numpy(float);cols.append(rank(rv,tv))
    return np.column_stack(cols)

def metrics(y,s,cut):
    p=s>=cut;pos=y.sum();neg=len(y)-pos;tp=((p==1)&(y==1)).sum();tn=((p==0)&(y==0)).sum()
    acc=(tp+tn)/len(y);base=max(y.mean(),1-y.mean())
    return {"bal":float(.5*(tp/pos+tn/neg)),"acc":float(acc),"edge":float(acc-base),"pred_up":float(p.mean())}

def main():
    data=app.load_search_frame(app.Config())
    if data.loc[data.source.eq("SEC_V26_SEAL"),"y"].notna().any():raise RuntimeError("V26 seal exposed")
    ref=data[data.source.eq("SEC_V24_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
    audit=data[data.source.eq("SEC_V24_SEAL")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
    dev=data[data.source.eq("SEC_V26_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
    ax=matrix(ref,audit);dx=matrix(dev,dev);ay=audit.y.astype(int).to_numpy();dy=dev.y.astype(int).to_numpy()
    h=dev.event_id.astype(str).map(lambda x:int(hashlib.sha256(x.encode()).hexdigest()[:16],16)%2).to_numpy()
    t=pd.to_datetime(dev.event_time_utc,utc=True).rank(method="first",pct=True).to_numpy()
    masks={"h0":h==0,"h1":h==1,"early":t<=.5,"late":t>.5}
    univ=[];stable=[];oa=np.empty_like(ax);od=np.empty_like(dx)
    for i,c in enumerate(FEATURES):
        aa=float(roc_auc_score(ay,ax[:,i]));da=float(roc_auc_score(dy,dx[:,i]))
        sign=1 if aa>.5 and da>.5 else (-1 if aa<.5 and da<.5 else 0)
        oa[:,i]=ax[:,i] if sign>=0 else 1-ax[:,i];od[:,i]=dx[:,i] if sign>=0 else 1-dx[:,i]
        a=aa if sign>=0 else 1-aa;d=da if sign>=0 else 1-da
        split={name:float(roc_auc_score(dy[m],od[m,i])) for name,m in masks.items()}
        if sign and min(a,d)>=.515:stable.append(i)
        univ.append({"column":c,"sign":sign,"audit_auc":a,"dev_auc":d,"split_auc":split})
    rng=np.random.default_rng(app.SEED+26);weights=[]
    for size in (1,2,3):
        for chosen in itertools.combinations(stable,size):
            w=np.zeros(len(FEATURES));w[list(chosen)]=1/size;weights.append(w)
    for _ in range(8000):
        size=int(rng.integers(2,min(7,len(stable)+1)));chosen=rng.choice(stable,size,replace=False)
        w=np.zeros(len(FEATURES));w[chosen]=rng.dirichlet(np.full(size,1.5));weights.append(w)
    heap=[];serial=0;seen=set()
    for w in weights:
        key=tuple(np.round(w,5))
        if key in seen:continue
        seen.add(key);a=oa@w;d=od@w;aa=float(roc_auc_score(ay,a));da=float(roc_auc_score(dy,d))
        split_auc={name:float(roc_auc_score(dy[m],d[m])) for name,m in masks.items()}
        if min(aa,da)<.525:continue
        components=[{"column":FEATURES[i],"sign":univ[i]["sign"],"weight":float(w[i])} for i in np.flatnonzero(w)]
        for cut in np.arange(.50,.901,.01):
            am=metrics(ay,a,cut);dm=metrics(dy,d,cut)
            splits={name:metrics(dy[m],d[m],cut) for name,m in masks.items()}
            passes=sum([am["bal"]>=.51,dm["bal"]>=.54,am["edge"]>=0,dm["edge"]>=.02,
                        aa>=.54,da>=.56])+sum(v["bal"]>=.49 for v in splits.values())
            floor_split=min(v["bal"] for v in splits.values());floor_auc=min(split_auc.values())
            score=8*min(am["bal"],dm["bal"])+5*min(am["edge"],dm["edge"])+2*min(aa,da)+floor_split+floor_auc
            row={"components":components,"cutoff":float(cut),"audit":{**am,"auc":aa},
                 "dev":{**dm,"auc":da},"split":splits,"split_auc":split_auc,"passes":passes,"score":float(score)}
            k=(passes,score,serial);serial+=1
            if len(heap)<1500:heapq.heappush(heap,(k,row))
            elif k>heap[0][0]:heapq.heapreplace(heap,(k,row))
    candidates=[x[1] for x in heap];candidates.sort(key=lambda r:(r["passes"],r["score"]),reverse=True)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"stable":[FEATURES[i] for i in stable],
            "univariate":sorted(univ,key=lambda r:min(r["audit_auc"],r["dev_auc"]),reverse=True),"candidates":candidates}
    path=ROOT/"cache"/"v26_rank.json";path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("UNIVARIATE")
    for r in result["univariate"][:25]:print(r)
    print("CANDIDATES")
    for r in candidates[:30]:print(json.dumps(r,ensure_ascii=False))
    print("RESULT="+str(path))

if __name__=="__main__":main()
