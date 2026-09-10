"""V27 source lineage, split isolation, and hidden-seal audit."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

import bio_news_30m_v3 as app

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"


def main()->None:
    def article_key(value)->str:
        if pd.isna(value):return ""
        text=str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():text=text[:-2]
        return str(int(text)) if text.isdigit() else text

    prior=pd.read_csv(DATA/"events_exact_v26.csv.gz",dtype={"ticker":str,"article_id":str})
    current=pd.read_csv(DATA/"events_exact_v27.csv.gz",dtype={"ticker":str,"article_id":str})
    labeled=pd.read_csv(DATA/"labeled_events_v27.csv.gz",dtype={"ticker":str})
    prior_ids=set(prior.event_id.astype(str))
    new=current[current.source.astype(str).str.contains("V27",na=False)].copy()
    new_ids=set(new.event_id.astype(str))
    prior_articles=set(prior.loc[
        prior.market.eq("KR")&prior.article_id.notna(),"article_id"
    ].map(article_key))
    new_articles=set(new.loc[
        new.market.eq("KR")&new.article_id.notna(),"article_id"
    ].map(article_key))
    search=app.load_search_frame(app.Config())
    seal_sources={"SEC_V27_SEAL","NAVER_NEWS_V27_SEAL"}
    dev_sources={"SEC_V27_DEV","NAVER_NEWS_V27_DEV"}
    seal=search[search.source.isin(seal_sources)]
    dev=search[search.source.isin(dev_sources)]
    report={
        "v26_event_rows":len(prior),"v27_event_rows":len(current),
        "v27_new_event_rows":len(new),
        "v27_sources_inside_v26":int(prior.source.astype(str).str.contains("V27",na=False).sum()),
        "prior_event_id_overlap":len(prior_ids&new_ids),
        "prior_kr_article_overlap":len(prior_articles&new_articles),
        "labeled_rows":len(labeled),
        "labeled_new_counts":{
            f"{market}|{source}":int(count)
            for (market,source),count in labeled[
                labeled.source.isin(seal_sources|dev_sources)
            ].groupby(["market","source"]).size().items()
        },
        "search_dev_y_visible":int(dev.y.notna().sum()),
        "search_seal_rows":len(seal),
        "search_seal_y_visible":int(seal.y.notna().sum()),
        "search_seal_return_visible":int(seal.fwd_ret_30m.notna().sum()),
        "dev_seal_event_overlap":len(
            set(dev.event_id.astype(str))&set(seal.event_id.astype(str))
        ),
    }
    checks={
        "v26_clean":report["v27_sources_inside_v26"]==0,
        "prior_event_disjoint":report["prior_event_id_overlap"]==0,
        "prior_article_disjoint":report["prior_kr_article_overlap"]==0,
        "seal_hidden":report["search_seal_y_visible"]==0 and report["search_seal_return_visible"]==0,
        "dev_visible":report["search_dev_y_visible"]>0,
        "dev_seal_disjoint":report["dev_seal_event_overlap"]==0,
        "seal_minimums":all(report["labeled_new_counts"].get(key,0)>=minimum for key,minimum in {
            "US|SEC_V27_SEAL":250,"KR|NAVER_NEWS_V27_SEAL":100,
        }.items()),
    }
    report["checks"]=checks;report["all_checks_pass"]=all(checks.values())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report["all_checks_pass"]:raise SystemExit(1)


if __name__=="__main__":main()
