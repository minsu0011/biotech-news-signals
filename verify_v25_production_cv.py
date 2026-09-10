"""Exact pre-freeze verification of the V25 production search path."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import bio_news_30m_v3 as app
import runtime_limits


def main()->None:
    cfg=app.Config();data=app.load_search_frame(cfg)
    seals={source for _,source in cfg.fixed_seal_sources}
    if data.loc[data.source.isin(seals),"y"].notna().any():
        raise RuntimeError("V25 seal outcome exposure detected")
    parts=[]
    for market in ("US","KR"):
        spec=app.model_specs(market)[0]
        parts.append(app.cv_market_predictions(data,market,spec,cfg))
    combined=pd.concat(parts,ignore_index=True)
    metrics=app.evaluate(combined,float(cfg.fixed_highconf_threshold),cfg.round_trip_cost)
    bootstrap=app.bootstrap_by_day(combined,float(cfg.fixed_highconf_threshold),
                                   cfg.round_trip_cost,nboot=2000)
    checks=app.cert_checks(metrics,bootstrap,cfg)
    result={"seal_outcomes_loaded":False,"runtime":runtime_limits.status(),
            "metrics":metrics,"bootstrap":bootstrap,"checks":checks,
            "all_checks_pass":all(row["pass"] for row in checks)}
    path=Path(__file__).resolve().parent/"cache"/"v25_production_cv.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print("RESULT="+str(path))


if __name__=="__main__":main()
