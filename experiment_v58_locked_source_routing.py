"""V58: fixed source-DGP routing across already nested outer-OOF experts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V58")))
CACHE=ROOT/"cache"
FILES={"SEC_NUMERIC":"v53_decoupled_oof.csv.gz",
       "US_NEWS_MULTILINGUAL":"v50_embedding_oof.csv.gz",
       "KR_DIVERSE_CONSENSUS":"v54_diverse_consensus_oof.csv.gz",
       "ECONOMIC_CONFIDENCE":"v48_xgb_expert_oof.csv.gz"}
ROUTES={"US_SEC":"SEC_NUMERIC","US_NEWS":"US_NEWS_MULTILINGUAL",
        "KR_NEWS":"KR_DIVERSE_CONSENSUS","KR_KIND":"KR_DIVERSE_CONSENSUS"}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True)
    frames={name:pd.read_csv(CACHE/file,compression="gzip",low_memory=False,
                             parse_dates=["event_time_utc"],dtype={"ticker":str})
            for name,file in FILES.items()}
    selected=frames["KR_DIVERSE_CONSENSUS"].copy()
    if selected.event_id.duplicated().any():raise RuntimeError("V58 duplicate event")
    aligned={name:frame.set_index("event_id").loc[selected.event_id] for name,frame in frames.items()}
    for name,frame in aligned.items():
        if not np.array_equal(frame.fold.to_numpy(),selected.fold.to_numpy()):
            raise RuntimeError(f"V58 fold mismatch: {name}")
    selected["routed_from"]=""
    for source,expert in ROUTES.items():
        mask=selected.source_family.eq(source).to_numpy()
        selected.loc[mask,"prob"]=aligned[expert].prob.to_numpy(float)[mask]
        selected.loc[mask,"routed_from"]=expert
    confidence=aligned["ECONOMIC_CONFIDENCE"]
    selected["high_conf"]=confidence.high_conf.to_numpy(bool)
    selected["confidence_signal"]=confidence.confidence_signal.to_numpy(float)
    selected["family"]="V58_LOCKED_SOURCE_DGP_ROUTING"
    if selected.routed_from.eq("").any():raise RuntimeError("V58 unrouted source")
    result=v44.summarize(selected)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation=("all routed predictions are canonical chronological outer OOF; routes and source identities are "
                "label-blind at inference; no per-row or current-fold model choice; one V48 inner-only confidence mask")
    comparison={"version":"V58","hypothesis":"LOCKED_SOURCE_DGP_ROUTING",
        "component_sha256":{name:v37.sha256(CACHE/file) for name,file in FILES.items()},
        "fixed_routes":ROUTES,"models":{"V58_LOCKED_SOURCE_DGP_ROUTING":result},
        "validation":validation}
    robustness={"version":"V58","status":status,"hypothesis":"LOCKED_SOURCE_DGP_ROUTING",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "fixed_routes":ROUTES,"seal_authorized":False}
    transfer={"version":"V58","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v58_locked_source_routing_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],
                      "gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
