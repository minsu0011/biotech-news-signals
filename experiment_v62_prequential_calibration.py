"""V62: source-specific Platt calibration fitted only on prior OOF folds."""
from __future__ import annotations
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits

ROOT=Path(__file__).resolve().parent;CACHE=ROOT/"cache"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V62")))
BASE=CACHE/"v58_locked_source_routing_oof.csv.gz"

def calibrate(train:pd.DataFrame,target:pd.DataFrame)->np.ndarray:
    if len(train)<150 or train.y.nunique()<2:return target.prob.to_numpy(float)
    x=logit(np.clip(train.prob.to_numpy(float),1e-5,1-1e-5)).reshape(-1,1)
    xt=logit(np.clip(target.prob.to_numpy(float),1e-5,1-1e-5)).reshape(-1,1)
    model=LogisticRegression(C=.25,class_weight="balanced",solver="liblinear",random_state=62)
    model.fit(x,train.y.astype(int),sample_weight=train.cluster_weight.to_numpy(float))
    return model.predict_proba(xt)[:,1]

def apply(frame:pd.DataFrame,source_specific:bool)->tuple[pd.DataFrame,list[dict]]:
    output=frame.copy();audits=[]
    for fold in sorted(frame.fold.unique()):
        history=frame[frame.fold<fold];target_all=frame[frame.fold.eq(fold)]
        groups=sorted(target_all.source_family.unique()) if source_specific else ["ALL"]
        for source in groups:
            target=target_all if source=="ALL" else target_all[target_all.source_family.eq(source)]
            train=history if source=="ALL" else history[history.source_family.eq(source)]
            probability=calibrate(train,target);positions=np.flatnonzero(output.event_id.isin(set(target.event_id)).to_numpy())
            lookup=dict(zip(target.event_id,probability));output.loc[positions,"prob"]=output.iloc[positions].event_id.map(lookup).to_numpy(float)
            output.loc[positions,"family"]="V62_SOURCE_PLATT" if source_specific else "V62_GLOBAL_PLATT"
            audits.append({"fold":int(fold),"source":source,"history_n":len(train),"target_n":len(target),
                "history_folds":sorted(int(value) for value in history.fold.unique()),"current_fold_labels_used":False,
                "fallback_identity":len(train)<150 or train.y.nunique()<2})
    return output,audits

def main()->None:
    runtime_limits.configure();OUT.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(BASE,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    source,audits=apply(base,True);global_frame,global_audits=apply(base,False)
    models={"SOURCE_PREQUENTIAL_PLATT":v44.summarize(source),"GLOBAL_PREQUENTIAL_PLATT":v44.summarize(global_frame),
            "IDENTITY_V58":v44.summarize(base)};result=models["SOURCE_PREQUENTIAL_PLATT"]
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation=("calibrators use only completed earlier outer-OOF folds; fold 1 is identity; fixed C=0.25; "
                "direction routes and economic confidence are frozen from V58")
    comparison={"version":"V62","hypothesis":"SOURCE_PREQUENTIAL_CALIBRATION_V1","base_sha256":v37.sha256(BASE),
        "fixed_selected":"SOURCE_PREQUENTIAL_PLATT","models":models,"source_audits":audits,
        "global_audits":global_audits,"validation":validation}
    robustness={"version":"V62","status":status,"hypothesis":"SOURCE_PREQUENTIAL_CALIBRATION_V1",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,"seal_authorized":False}
    transfer={"version":"V62","seal_outcomes_loaded":False,"selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],"selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    source.to_csv(CACHE/"v62_prequential_calibration_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))

if __name__=="__main__":main()
