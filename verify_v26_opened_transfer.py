"""One-version-back rolling audit of the exact V26 production classes."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import bio_news_30m_v3 as app
import runtime_limits

TARGET={"US":909.0,"KR":450.0}

def main():
    cfg=app.Config();data=app.load_search_frame(cfg)
    if data.loc[data.source.isin({"SEC_V26_SEAL","NAVER_NEWS_V26_SEAL"}),"y"].notna().any():raise RuntimeError("V26 seal exposed")
    # US: V24 DEV reference -> already-opened V24 SEAL audit.
    us_ref=data[data.source.eq("SEC_V24_DEV")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
    us=data[data.source.eq("SEC_V24_SEAL")&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
    um=app.make_model(app.model_specs("US")[0]);um.fit(app.model_frame(us_ref),us_ref.y.astype(int))
    up=um.predict_proba(app.model_frame(us))[:,1]
    # KR rolling architecture: opened V24 prior + grouped-OOF opened V25 target.
    kr_prior=data[data.source.isin({"NAVER_NEWS_V24_DEV","NAVER_NEWS_V24_SEAL"})&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
    kr=data[data.source.isin({"NAVER_NEWS_V25_DEV","NAVER_NEWS_V25_SEAL"})&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
    km=app.make_model(app.model_specs("KR")[0]);kf=app.model_frame(kr)
    km.fit_with_prior(kf,kr.y.astype(int),app.model_frame(kr_prior),kr_prior.y.astype(int))
    kp=km.crossfit_output(kf)[0]
    parts=[]
    for market,frame,p in (("US",us,up),("KR",kr,kp)):
        z=frame[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
        z["prob"]=p;z["eval_weight"]=TARGET[market]/len(z);parts.append(z)
    combined=pd.concat(parts,ignore_index=True);metrics=app.evaluate(combined,.925,.002)
    bootstrap=app.bootstrap_by_day(combined,.925,.002,nboot=2000)
    checks={"balanced_accuracy":metrics["balanced_accuracy"]>=.54,"auc":metrics["auc"]>=.56,
      "edge":metrics["edge_vs_naive"]>=.02,"coverage":metrics["highconf_coverage"]>=.15,
      "highconf_accuracy":metrics["highconf_accuracy"]>=.60,"highconf_net":metrics["strategy_mean_signed_net"]>0,
      "bootstrap":bootstrap["balanced_accuracy_lower95"]>=.50,
      "US_bal":metrics["by_market"]["US"]["balanced_accuracy"]>=.51,
      "KR_bal":metrics["by_market"]["KR"]["balanced_accuracy"]>=.51}
    result={"v26_seal_outcomes_loaded":False,"audit":"rolling opened V24/V25",
            "runtime":runtime_limits.status(),"metrics":metrics,"bootstrap":bootstrap,
            "checks":checks,"all_checks_pass":all(checks.values())}
    path=Path(__file__).resolve().parent/"cache"/"v26_opened_transfer.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2));print("RESULT="+str(path))

if __name__=="__main__":main()
