"""KR rolling architecture audit: V24 -> opened V25 target OOF."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import bio_news_30m_v3 as app
import runtime_limits

ROOT=Path(__file__).resolve().parent
SPECS={f"cat{d}":{"kind":"native_cat","iterations":350,"depth":d,"cat_cols":["event_type","ticker"]} for d in (2,4,6)}
SPECS.update({f"lgb{l}":{"kind":"lgb","n_estimators":350,"num_leaves":l,"cat_cols":["event_type","ticker"]} for l in (3,5,9,15)})
SPECS.update({"logit005":{"kind":"logit","C":.005,"class_weight":"balanced"},"logit02":{"kind":"logit","C":.02,"class_weight":"balanced"}})

def main():
 data=app.load_search_frame(app.Config())
 if data.loc[data.source.eq("NAVER_NEWS_V26_SEAL"),"y"].notna().any():raise RuntimeError("V26 seal exposed")
 prior=data[data.source.isin({"NAVER_NEWS_V24_DEV","NAVER_NEWS_V24_SEAL"})&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
 target=data[data.source.isin({"NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL"})&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
 groups=target.event_id.astype(str).str.rsplit(":",n=1).str[-1]
 splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=app.SEED);splits=list(splitter.split(target,target.y.astype(int),groups))
 rows=[];cache=[];y=target.y.astype(int).to_numpy();up=float(y.mean())
 for name,spec in SPECS.items():
  for repeat in ((1,3,6) if name.startswith(("cat","lgb")) else (1,3)):
   p=np.full(len(target),np.nan)
   for tr,va in splits:
    train=pd.concat([prior]+[target.iloc[tr]]*repeat,ignore_index=True,sort=False)
    labels=np.concatenate([prior.y.astype(int).to_numpy()]+[target.iloc[tr].y.astype(int).to_numpy()]*repeat)
    model=app.make_model(spec);model.fit(app.model_frame(train),labels);p[va]=model.predict_proba(app.model_frame(target.iloc[va]))[:,1]
   auc=float(roc_auc_score(y,p));choices=[]
   for cut in np.arange(.25,.751,.005):
    pred=p>=cut;acc=float((pred==y).mean());bal=float(balanced_accuracy_score(y,pred));edge=acc-max(up,1-up)
    choices.append({"cutoff":float(cut),"accuracy":acc,"balanced_accuracy":bal,"edge":edge,"auc":auc,
                    "passes":sum([bal>=.54,edge>=.02,auc>=.56]),"score":float(8*bal+5*edge+3*auc)})
   choices.sort(key=lambda r:(r["passes"],r["score"]),reverse=True)
   full=f"{name}_r{repeat}";rows.append({"name":full,"spec":spec,"target_repeat":repeat,"choices":choices[:30]})
   z=target[["event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"]].copy();z["model_name"]=full;z["probability"]=p;cache.append(z)
   b=choices[0];print(f"[{full}] pass={b['passes']}/3 bal={b['balanced_accuracy']:.4f} auc={auc:.4f} edge={b['edge']:.4f} cut={b['cutoff']:.3f}",flush=True)
 rows.sort(key=lambda r:(r["choices"][0]["passes"],r["choices"][0]["score"]),reverse=True)
 result={"v26_seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"target_n":len(target),"models":rows}
 pd.concat(cache).to_parquet(ROOT/"cache"/"v26_rolling_kr_oof.parquet",index=False)
 path=ROOT/"cache"/"v26_rolling_kr.json";path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");print("RESULT="+str(path))

if __name__=="__main__":main()
