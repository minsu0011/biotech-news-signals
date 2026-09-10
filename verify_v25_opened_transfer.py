"""Production-class audit on the already-opened V24 seal."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

import bio_news_30m_v3 as app
import runtime_limits

TARGET={"US":295.0,"KR":521.0}
SOURCES={"US":("SEC_V24_DEV","SEC_V24_SEAL"),
         "KR":("NAVER_NEWS_V24_DEV","NAVER_NEWS_V24_SEAL")}


def main()->None:
    cfg=app.Config();data=app.load_search_frame(cfg);parts=[]
    if data.loc[data.source.isin({"SEC_V25_SEAL","NAVER_NEWS_V25_SEAL"}),"y"].notna().any():
        raise RuntimeError("V25 seal outcome exposure detected")
    for market,(ref_source,audit_source) in SOURCES.items():
        ref=data[data.source.eq(ref_source)&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
        audit=data[data.source.eq(audit_source)&data.y.notna()].copy().sort_values("event_time_utc").reset_index(drop=True)
        model=app.make_model(app.model_specs(market)[0]);model.fit(app.model_frame(ref),ref.y.astype(int))
        probability=model.predict_proba(app.model_frame(audit))[:,1]
        z=audit[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
        z["prob"]=probability;z["eval_weight"]=TARGET[market]/len(z);parts.append(z)
    combined=pd.concat(parts,ignore_index=True)
    metrics=app.evaluate(combined,.925,.002)
    bootstrap=app.bootstrap_by_day(combined,.925,.002,nboot=2000)
    performance_checks={
        "balanced_accuracy":metrics["balanced_accuracy"]>=.54,
        "auc":metrics["auc"]>=.56,"edge":metrics["edge_vs_naive"]>=.02,
        "coverage":metrics["highconf_coverage"]>=.15,
        "highconf_accuracy":metrics["highconf_accuracy"]>=.60,
        "highconf_net":metrics["strategy_mean_signed_net"]>0,
        "bootstrap":bootstrap["balanced_accuracy_lower95"]>=.50,
        "US_bal":metrics["by_market"]["US"]["balanced_accuracy"]>=.51,
        "KR_bal":metrics["by_market"]["KR"]["balanced_accuracy"]>=.51,
    }
    result={"v25_seal_outcomes_loaded":False,"audit":"opened V24 seal",
            "runtime":runtime_limits.status(),"metrics":metrics,"bootstrap":bootstrap,
            "performance_checks":performance_checks,
            "all_performance_checks_pass":all(performance_checks.values())}
    path=Path(__file__).resolve().parent/"cache"/"v25_opened_transfer.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2));print("RESULT="+str(path))


if __name__=="__main__":main()
