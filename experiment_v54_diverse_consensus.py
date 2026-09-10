"""V54: fixed consensus across three materially different direction experts.

The equal weights and V48 high-confidence policy are declared before this run.
Every component is an outer-OOF prediction whose fitting and confidence policy
were selected inside its own chronological training fold.  Current-fold labels
are never used to route, weight, or threshold the consensus.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V54")))
CACHE=ROOT/"cache"
BASE_FILES={
    "opportunity_direction":"v38_opportunity_oof.csv.gz",
    "event_form_direction":"v48_xgb_expert_oof.csv.gz",
    "fixed_numeric_direction":"v53_decoupled_oof.csv.gz",
}


def load_components()->tuple[pd.DataFrame,dict[str,pd.DataFrame]]:
    frames={name:pd.read_csv(CACHE/file,compression="gzip",low_memory=False,
                             parse_dates=["event_time_utc"],dtype={"ticker":str})
            for name,file in BASE_FILES.items()}
    authority=frames["fixed_numeric_direction"].copy()
    if authority.event_id.duplicated().any():
        raise RuntimeError("V54 duplicate authority event_id")
    for name,frame in frames.items():
        if frame.event_id.duplicated().any() or set(frame.event_id)!=set(authority.event_id):
            raise RuntimeError(f"V54 component event mismatch: {name}")
        aligned=frame.set_index("event_id").loc[authority.event_id]
        authority[f"prob_{name}"]=aligned.prob.to_numpy(float)
        authority[f"high_{name}"]=aligned.high_conf.to_numpy(bool)
        authority[f"confidence_{name}"]=aligned.confidence_signal.to_numpy(float)
        if not np.array_equal(aligned.fold.to_numpy(),authority.fold.to_numpy()):
            raise RuntimeError(f"V54 fold mismatch: {name}")
    return authority,frames


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True)
    selected,frames=load_components()
    probability_columns=[f"prob_{name}" for name in BASE_FILES]
    # Fixed, non-fitted consensus: structurally distinct experts each receive
    # exactly one third.  V48's independently inner-selected opportunity mask
    # is preserved; no V54 outcome is consulted.
    selected["prob"]=selected[probability_columns].mean(axis=1)
    selected["high_conf"]=selected["high_event_form_direction"].astype(bool)
    selected["confidence_signal"]=selected["confidence_event_form_direction"].astype(float)
    selected["family"]="V54_FIXED_DIVERSE_CONSENSUS"
    result=v44.summarize(selected)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    component_diagnostics={}
    for name in BASE_FILES:
        component=selected.copy()
        component["prob"]=component[f"prob_{name}"]
        component["high_conf"]=component[f"high_{name}"].astype(bool)
        component_diagnostics[name]={"metrics":v36.summary(component)}
    validation=("fixed equal weights; cached component predictions are chronological outer OOF; "
                "each base fit/policy is inner-only; current-fold outcomes are unused")
    comparison={"version":"V54","hypothesis":"DIVERSE_DIRECTION_CONSENSUS",
        "base_cache_sha256":{name:v37.sha256(CACHE/file) for name,file in BASE_FILES.items()},
        "declared_weights":{name:1/3 for name in BASE_FILES},
        "confidence_policy":"V48 independently nested event-form opportunity mask",
        "models":{"V54_FIXED_DIVERSE_CONSENSUS":result,**component_diagnostics},
        "validation":validation}
    robustness={"version":"V54","status":status,"hypothesis":"DIVERSE_DIRECTION_CONSENSUS",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "declared_weights":comparison["declared_weights"],"validation":validation,
        "seal_authorized":False}
    transfer={"version":"V54","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v54_diverse_consensus_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],
                      "gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
