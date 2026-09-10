"""Exact V26 production-path parity and pre-freeze checks."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import bio_news_30m_v3 as app
import runtime_limits

def main():
    cfg=app.Config();data=app.load_search_frame(cfg)
    if data.loc[data.source.isin({"SEC_V26_SEAL","NAVER_NEWS_V26_SEAL"}),"y"].notna().any():raise RuntimeError("V26 seal exposed")
    parts=[app.cv_market_predictions(data,m,app.model_specs(m)[0],cfg) for m in ("US","KR")]
    combined=pd.concat(parts,ignore_index=True);metrics=app.evaluate(combined,.925,.002)
    bootstrap=app.bootstrap_by_day(combined,.925,.002,nboot=2000)
    performance={"balanced_accuracy":metrics["balanced_accuracy"]>=.54,"auc":metrics["auc"]>=.56,
      "edge":metrics["edge_vs_naive"]>=.02,"coverage":metrics["highconf_coverage"]>=.15,
      "highconf_accuracy":metrics["highconf_accuracy"]>=.60,"highconf_net":metrics["strategy_mean_signed_net"]>0,
      "bootstrap":bootstrap["balanced_accuracy_lower95"]>=.50,
      "US_bal":metrics["by_market"]["US"]["balanced_accuracy"]>=.51,
      "KR_bal":metrics["by_market"]["KR"]["balanced_accuracy"]>=.51}
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),"metrics":metrics,
            "bootstrap":bootstrap,"performance_checks":performance,"all_performance_checks_pass":all(performance.values())}
    path=Path(__file__).resolve().parent/"cache"/"v26_production_cv.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2));print("RESULT="+str(path))

if __name__=="__main__":main()
