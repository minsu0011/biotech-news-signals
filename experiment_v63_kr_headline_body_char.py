"""V63: separate Korean headline/body character experts with fixed fusion."""
from __future__ import annotations
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits

ROOT=Path(__file__).resolve().parent;DATA=ROOT/"data";CACHE=ROOT/"cache"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V63")));BASE=CACHE/"v58_locked_source_routing_oof.csv.gz"

class CharExpert:
    def __init__(self,column:str):
        self.column=column;self.vector=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=3,max_df=.997,
            max_features=35000,sublinear_tf=True,dtype=np.float32)
        self.model=LogisticRegression(C=.5,class_weight="balanced",solver="liblinear",max_iter=300,random_state=63)
    def fit(self,frame):
        self.model.fit(self.vector.fit_transform(frame[self.column].fillna("").astype(str)),frame.y.astype(int),
                       sample_weight=frame.cluster_weight.to_numpy(float));return self
    def predict(self,frame):return self.model.predict_proba(self.vector.transform(frame[self.column].fillna("").astype(str)))[:,1]

def main()->None:
    runtime_limits.configure();OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1);data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    base=pd.read_csv(BASE,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    selected=base.copy();head_component=base.copy();body_component=base.copy();combined_component=base.copy();audits=[];base_map=base.set_index("event_id")
    for train,valid,fold in v37.chronological_folds(data[data.market.eq("KR")].copy()):
        train_news=train[train.source_family.eq("KR_NEWS")];target=valid[valid.source_family.eq("KR_NEWS")]
        head=CharExpert("headline").fit(train_news).predict(target);body=CharExpert("body").fit(train_news).predict(target)
        semantic=expit(.6*logit(np.clip(head,1e-5,1-1e-5))+.4*logit(np.clip(body,1e-5,1-1e-5)))
        p_base=base_map.loc[target.event_id,"prob"].to_numpy(float)
        blend=expit(.5*logit(np.clip(p_base,1e-5,1-1e-5))+.5*logit(np.clip(semantic,1e-5,1-1e-5)))
        positions=np.flatnonzero(base.event_id.isin(set(target.event_id)).to_numpy())
        for frame,prob,name in ((selected,blend,"V63_KR_CHAR_BLEND"),(head_component,head,"V63_HEADLINE_CHAR"),
                                (body_component,body,"V63_BODY_CHAR"),(combined_component,semantic,"V63_SEPARATE_CHAR")):
            lookup=dict(zip(target.event_id,prob));frame.loc[positions,"prob"]=frame.iloc[positions].event_id.map(lookup).to_numpy(float);frame.loc[positions,"family"]=name
        audits.append({"fold":fold,"train_KR_NEWS_n":len(train_news),"valid_KR_NEWS_n":len(target),
            "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min(),
            "fixed_semantic_weights":{"headline":.6,"body":.4},"fixed_base_blend":{"V58":.5,"separate_char":.5},
            "current_fold_labels_used":False})
    models={"V63_FIXED_BLEND":v44.summarize(selected),"HEADLINE_CHAR":v44.summarize(head_component),
            "BODY_CHAR":v44.summarize(body_component),"SEPARATE_CHAR":v44.summarize(combined_component)}
    result=models["V63_FIXED_BLEND"];status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation=("headline and body vocabularies/models are fit separately on each outer KR training fold; fixed fusion; "
                "V58 confidence is frozen; no current-fold or seal labels select weights")
    comparison={"version":"V63","hypothesis":"KR_NEWS_HEADLINE_BODY_CHAR_V1","base_sha256":v37.sha256(BASE),
        "fixed_selected":"V63_FIXED_BLEND","models":models,"fold_audits":audits,"validation":validation}
    robustness={"version":"V63","status":status,"hypothesis":"KR_NEWS_HEADLINE_BODY_CHAR_V1",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,"seal_authorized":False}
    transfer={"version":"V63","seal_outcomes_loaded":False,"selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],"selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v63_kr_headline_body_char_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))

if __name__=="__main__":main()
