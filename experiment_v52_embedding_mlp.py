"""V52 material hypothesis: nonlinear GPU head over frozen E5 embeddings."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v44_microstructure as v44
import experiment_v50_embeddings as v50
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V52")))
CACHE=ROOT/"cache"
SEED=20260827


class EmbeddingMLP(nn.Module):
    def __init__(self,dimension,hidden,dropout):
        super().__init__();self.network=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),
            nn.LayerNorm(hidden),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x):return self.network(x).squeeze(1)


def set_seed(offset=0):
    seed=SEED+offset;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def train_predict(train,target,matrix,hidden,dropout,weight_decay,offset):
    set_seed(offset);device=torch.device("cuda");index=train.embedding_index.to_numpy(int)
    x=torch.from_numpy(matrix[index]).float();y=torch.from_numpy(train.y.to_numpy(np.float32))
    w=torch.from_numpy(train.cluster_weight.to_numpy(np.float32));model=EmbeddingMLP(x.shape[1],hidden,dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=weight_decay)
    generator=torch.Generator().manual_seed(SEED+offset)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y,w),batch_size=256,
        shuffle=True,generator=generator,num_workers=0,pin_memory=True)
    model.train()
    for _ in range(35):
        for xb,yb,wb in loader:
            xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True);wb=wb.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(
                model(xb),yb,weight=wb,reduction="sum")/wb.sum();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
    model.eval();target_x=torch.from_numpy(matrix[target.embedding_index.to_numpy(int)]).float()
    pieces=[]
    with torch.inference_mode():
        for start in range(0,len(target_x),512):pieces.append(torch.sigmoid(
            model(target_x[start:start+512].to(device))).cpu().numpy())
    return np.concatenate(pieces)


CONFIGS={"mlp128":(128,.20,1e-3),"mlp256":(256,.35,5e-3)}


def market_oof(frame,market,matrix):
    selected=[];components=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        inner_train,inner_valid=v37.split_inner(train);numeric_inner=v38.fit_components(inner_train,inner_valid,.006)
        embedding_models={name:train_predict(inner_train,inner_valid,matrix,*config,fold*10+i)
                          for i,(name,config) in enumerate(CONFIGS.items())}
        policy=v50.choose_policy(inner_valid.reset_index(drop=True),embedding_models,numeric_inner)
        config=CONFIGS[policy["embedding_key"]];embedding=train_predict(train,valid,matrix,*config,fold*100)
        numeric=v38.fit_components(train,valid,.006);prob,high,confidence=v50.apply_policy(policy,embedding,numeric)
        seen=set(train.ticker.astype(str));selected.append(v38.prediction_frame(
            valid,prob,high,confidence,fold,"V52_INNER_SELECTED",seen))
        component_high=v38.rank_against(np.abs(embedding_models[policy["embedding_key"]]-.5),
                                         np.abs(embedding-.5))>=.80
        components.append(v38.prediction_frame(valid,embedding,component_high,np.abs(embedding-.5),fold,
                                                 policy["embedding_key"],seen))
        audit={key:value for key,value in policy.items() if key!="references"}
        audit.update({"fold":fold,"market":market,"train_n":len(train),"inner_train_n":len(inner_train),
                      "inner_valid_n":len(inner_valid),"valid_n":len(valid),
                      "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit)
        print(f"[V52 MLP] {market} fold={fold} head={policy['embedding_key']} "
              f"blend={policy['blend_name']} cutoff={policy['direction_cutoff']:.2f} "
              f"checks={policy['checks']}/6",flush=True)
    return pd.concat(selected,ignore_index=True),pd.concat(components,ignore_index=True),audits


def main()->None:
    runtime_limits.configure()
    if not torch.cuda.is_available():raise RuntimeError("V52 CUDA unavailable")
    OUT.mkdir(exist_ok=True);path=DATA/"dev_contract_v36_labeled.csv.gz";embedding_path=CACHE/"v50_multilingual_e5_embeddings.npz"
    status=json.loads((CACHE/"V50_EMBEDDING_STATUS.json").read_text(encoding="utf-8"))
    if v37.sha256(embedding_path)!=status["embedding_sha256"]:raise RuntimeError("V52 embedding hash mismatch")
    loaded=np.load(embedding_path,allow_pickle=True);ids=loaded["event_id"].astype(str)
    headline=loaded["headline"].astype(np.float32);semantic=loaded["semantic"].astype(np.float32)
    matrix=np.concatenate([headline,semantic],axis=1);mapping={event_id:index for index,event_id in enumerate(ids)}
    data=pd.read_csv(path,compression="gzip",low_memory=False,dtype={"ticker":str},parse_dates=["event_time_utc"])
    data["embedding_index"]=data.event_id.astype(str).map(mapping).astype(int)
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    selected=[];components=[];selection={}
    for market in ("US","KR"):
        first,second,audits=market_oof(data[data.market.eq(market)].copy(),market,matrix)
        selected.append(first);components.append(second);selection[market]=audits
    selected_frame=pd.concat(selected,ignore_index=True);component_frame=pd.concat(components,ignore_index=True)
    result=v44.summarize(selected_frame);component=v44.summarize(component_frame)
    status_name="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    comparison={"version":"V52","hypothesis":"NONLINEAR_EMBEDDING_HEAD","dev_sha256":v37.sha256(path),
        "embedding_sha256":v37.sha256(embedding_path),"device":torch.cuda.get_device_name(0),
        "models":{"V52_INNER_SELECTED":result,"MLP_component":component},"selection_audits":selection}
    robustness={"version":"V52","status":status_name,"hypothesis":"NONLINEAR_EMBEDDING_HEAD",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "MLP_component":component,"selection_audits":selection,"seal_authorized":False}
    transfer={"version":"V52","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],
        "method":"frozen embeddings; GPU MLP weights and fusion fitted inside each fold only"}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json");v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected_frame.to_csv(CACHE/"v52_embedding_mlp_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status_name,"metrics":result["metrics"],"gate":result["research_gate"]},
                     indent=2,default=str))


if __name__=="__main__":main()
