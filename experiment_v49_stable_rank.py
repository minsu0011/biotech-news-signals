"""V49 material hypothesis: stable-sign empirical-rank sparse logistic."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V49")))
CACHE=ROOT/"cache"
COST=.002


def numeric_frame(frame,extra):
    base=app.model_frame(frame).reset_index(drop=True)
    numeric=[column for column in base if column not in v36.CAT_COLUMNS]
    output=base[numeric].apply(pd.to_numeric,errors="coerce")
    additions=frame[extra].apply(pd.to_numeric,errors="coerce").reset_index(drop=True)
    return pd.concat([output,additions],axis=1)


class StableRankLogit:
    def __init__(self,k:int,c:float,extra:list[str]):self.k=k;self.c=c;self.extra=extra

    def transform(self,frame):
        values=numeric_frame(frame,self.extra);output=np.empty((len(values),len(self.columns)),dtype=np.float32)
        for position,column in enumerate(self.columns):
            raw=pd.to_numeric(values[column],errors="coerce").to_numpy(float);reference=self.references[column]
            rank=np.searchsorted(reference,raw,side="right")/max(1,len(reference));rank[~np.isfinite(raw)]=.5
            output[:,position]=rank-.5
        return output

    @staticmethod
    def correlation(x,y,w):
        x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float);valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)
        if valid.sum()<20:return 0.
        x=x[valid];y=y[valid];w=w[valid];w=w/w.sum();mx=np.sum(w*x);my=np.sum(w*y)
        covariance=np.sum(w*(x-mx)*(y-my));den=np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2))
        return float(covariance/den) if den>0 else 0.

    def fit(self,frame,label):
        values=numeric_frame(frame,self.extra);self.columns=list(values)
        self.references={column:np.sort(pd.to_numeric(values[column],errors="coerce").dropna().to_numpy(float))
                         for column in self.columns}
        matrix=self.transform(frame);y=np.asarray(label,int);w=frame.cluster_weight.to_numpy(float)
        split=max(100,len(frame)//2);scores=[]
        for position,column in enumerate(self.columns):
            first=self.correlation(matrix[:split,position],y[:split],w[:split])
            second=self.correlation(matrix[split:,position],y[split:],w[split:])
            stable=np.sign(first)==np.sign(second) and first!=0 and second!=0
            score=min(abs(first),abs(second)) if stable else 0.
            scores.append((score,position,column,first,second))
        scores.sort(reverse=True);selected=[position for score,position,*_ in scores if score>0][:self.k]
        if len(selected)<min(5,self.k):
            overall=sorted(((abs(self.correlation(matrix[:,position],y,w)),position)
                            for position in range(matrix.shape[1])),reverse=True)
            selected=[position for _,position in overall[:self.k]]
        self.selected=np.asarray(selected,int);self.selection=[{
            "column":scores_item[2],"first_corr":scores_item[3],"second_corr":scores_item[4],
            "stable_score":scores_item[0]} for scores_item in scores if scores_item[1] in set(selected)]
        self.model=LogisticRegression(C=self.c,penalty="l1",solver="liblinear",max_iter=1000,
                                      random_state=v36.SEED)
        self.model.fit(matrix[:,self.selected],y,sample_weight=w);return self

    def predict(self,frame):return self.model.predict_proba(self.transform(frame)[:,self.selected])[:,1]


def fit_pair(train,target,k,c,extra):
    direction=StableRankLogit(k,c,extra).fit(train,train.y)
    opportunity=StableRankLogit(k,c,extra).fit(train,train.fwd_ret_30m.abs()>=.006)
    return {"direction":direction.predict(target),"opportunity":opportunity.predict(target)},\
           {"direction_features":direction.selection,"opportunity_features":opportunity.selection}


def choose_policy(frame,models):
    rows=[];y=frame.y.to_numpy(int);returns=frame.fwd_ret_30m.to_numpy(float);weight=frame.cluster_weight.to_numpy(float)
    for (k,c),(parts,_) in models.items():
        raw=parts["direction"]
        for cutoff in np.arange(.44,.561,.02):
            margin=v38.rank_self(np.abs(raw-cutoff));opportunity=v38.rank_self(parts["opportunity"])
            confidence=.55*margin+.45*opportunity;metrics=v37.metric(frame,raw,float(cutoff),True)
            floor=v37.source_floor(frame,raw,float(cutoff))
            for quantile in (.75,.80,.85):
                threshold=float(np.quantile(confidence,quantile));high=confidence>=threshold;prediction=raw>=cutoff
                accuracy=float(np.average(prediction[high]==y[high],weights=weight[high]))
                net=float(np.average(np.where(prediction[high],1.,-1.)*returns[high]-COST,weights=weight[high]))
                checks=sum((metrics["balanced_accuracy"]>=.52,metrics["auc"]>=.52,
                            metrics["edge"]>=0,accuracy>=.55,net>0,floor>=.49))
                score=(2*metrics["balanced_accuracy"]+1.5*metrics["auc"]+metrics["edge"]+
                       .4*floor+.35*accuracy+8*max(-.01,min(.01,net)))
                rows.append({"k":k,"C":c,"direction_cutoff":float(cutoff),
                    "confidence_quantile":quantile,"confidence_threshold":threshold,"checks":checks,
                    "score":float(score),"inner_metrics":metrics,"source_floor":floor,
                    "inner_hc_accuracy":accuracy,"inner_hc_net":net,
                    "references":{"margin":np.abs(raw-cutoff),"opportunity":parts["opportunity"]}})
    rows.sort(key=lambda row:(row["checks"],row["score"]),reverse=True);return rows[0]


def apply_policy(policy,parts):
    raw=parts["direction"];cutoff=float(policy["direction_cutoff"]);ref=policy["references"]
    confidence=(.55*v38.rank_against(ref["margin"],np.abs(raw-cutoff))+
                .45*v38.rank_against(ref["opportunity"],parts["opportunity"]))
    high=confidence>=float(policy["confidence_threshold"])
    return app.shift_probability(raw,cutoff),high,confidence


def market_oof(frame,market,extra):
    selected=[];components=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);models={}
        for k in (10,25,50):
            for c in (.03,.10,.30):models[(k,c)]=fit_pair(inner_train,inner_valid,k,c,extra)
        policy=choose_policy(inner_valid.reset_index(drop=True),models);spec=(int(policy["k"]),float(policy["C"]))
        outer,features=fit_pair(train,valid,*spec,extra);prob,high,confidence=apply_policy(policy,outer)
        seen=set(train.ticker.astype(str));selected.append(v38.prediction_frame(
            valid,prob,high,confidence,fold,"V49_INNER_SELECTED",seen))
        raw=outer["direction"];component_high=v38.rank_against(
            np.abs(models[spec][0]["direction"]-.5),np.abs(raw-.5))>=.80
        components.append(v38.prediction_frame(valid,raw,component_high,np.abs(raw-.5),fold,"stable_rank",seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),"selected_features":features,
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V49 RANK] {market} fold={fold} k={spec[0]} C={spec[1]:.2f} "
              f"cutoff={policy['direction_cutoff']:.2f} checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(components,ignore_index=True),audits


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True);path=DATA/"dev_contract_v36_labeled.csv.gz"
    paths=[CACHE/"v41_sector_features.csv.gz",CACHE/"v44_microstructure_features.csv.gz",
           CACHE/"v46_article_features.csv.gz"]
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    extra=[]
    for feature_path in paths:
        feature=pd.read_csv(feature_path,compression="gzip",low_memory=False)
        columns=[column for column in feature if column!="event_id"];extra.extend(columns)
        data=data.merge(feature,on="event_id",how="left",validate="one_to_one")
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];components=[];selection={}
    for market in ("US","KR"):
        first,second,audits=market_oof(data[data.market.eq(market)].copy(),market,extra)
        selected.append(first);components.append(second);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);component_frame=pd.concat(components,ignore_index=True)
    result=v44.summarize(selected_frame);component=v44.summarize(component_frame)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V49","hypothesis":"STABLE_RANK_LOGIT","dev_sha256":v37.sha256(path),
        "feature_sha256":{file.name:v37.sha256(file) for file in paths},
        "models":{"V49_INNER_SELECTED":result,"stable_rank_component":component},"selection_audits":selection}
    robustness={"version":"V49","status":status,"hypothesis":"STABLE_RANK_LOGIT",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "component":component,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V49","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"fold-fitted empirical ranks; stable-sign selection across chronological train halves; sparse L1"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v49_stable_rank_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],"gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
