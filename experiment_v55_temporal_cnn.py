"""V55: market-specific temporal CNN over raw causal one-minute bar sequences."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V55")))
CACHE=ROOT/"cache"
SEQUENCE_PATH=CACHE/"v55_bar_sequences.npz"
BASE_PATH=CACHE/"v54_diverse_consensus_oof.csv.gz"
MAX_EPOCHS=35
PATIENCE=6
BATCH_SIZE=256
BLEND_WEIGHT=.50


class ResidualBlock(nn.Module):
    def __init__(self,channels:int,dilation:int)->None:
        super().__init__();padding=2*dilation
        self.net=nn.Sequential(
            nn.Conv1d(channels,channels,5,padding=padding,dilation=dilation),
            nn.GroupNorm(6,channels),nn.GELU(),nn.Dropout(.10),
            nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),
            nn.GroupNorm(6,channels),nn.GELU(),
        )
    def forward(self,x:torch.Tensor)->torch.Tensor:return x+self.net(x)


class TemporalCNN(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(6,48,5,padding=2),nn.GroupNorm(6,48),nn.GELU())
        self.blocks=nn.Sequential(*(ResidualBlock(48,d) for d in (1,2,4,8)))
        self.head=nn.Sequential(nn.Linear(48*3,96),nn.GELU(),nn.Dropout(.20),
                                nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        mask=x[:,:,5].clamp(0,1);z=self.blocks(self.stem(x.transpose(1,2)))
        denominator=mask.sum(1,keepdim=True).clamp_min(1.)
        mean=(z*mask[:,None,:]).sum(2)/denominator
        maximum=z.masked_fill(~mask[:,None,:].bool(),-1e4).amax(2)
        positions=torch.arange(mask.shape[1],device=x.device)[None,:].expand_as(mask)
        last_index=positions.masked_fill(~mask.bool(),0).amax(1)
        last=z.gather(2,last_index[:,None,None].expand(-1,z.shape[1],1)).squeeze(2)
        return self.head(torch.cat([mean,maximum,last],dim=1)).squeeze(1)


def seed_all(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def normalizer(x:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    observed=x[:,:,5]>0;mean=np.zeros(6,dtype=np.float32);std=np.ones(6,dtype=np.float32)
    for channel in range(5):
        values=x[:,:,channel][observed];values=values[np.isfinite(values)]
        if len(values):mean[channel]=np.mean(values);std[channel]=max(np.std(values),1e-3)
    return mean,std


def transform(x:np.ndarray,mean:np.ndarray,std:np.ndarray)->np.ndarray:
    output=x.copy();mask=output[:,:,5]>0
    for channel in range(5):
        values=(output[:,:,channel]-mean[channel])/std[channel]
        output[:,:,channel]=np.where(mask,np.clip(values,-8,8),0.)
    output[:,:,5]=mask.astype(np.float32);return output


def loader(x:np.ndarray,y:np.ndarray,w:np.ndarray,shuffle:bool)->DataLoader:
    dataset=TensorDataset(torch.from_numpy(x.astype(np.float32)),
                          torch.from_numpy(y.astype(np.float32)),
                          torch.from_numpy(w.astype(np.float32)))
    return DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=0,
                      pin_memory=True,drop_last=False)


@torch.inference_mode()
def predict(model:nn.Module,x:np.ndarray,device:torch.device)->np.ndarray:
    model.eval();parts=[]
    dummy_y=np.zeros(len(x),np.float32);dummy_w=np.ones(len(x),np.float32)
    for batch,_,_ in loader(x,dummy_y,dummy_w,False):
        with torch.amp.autocast(device_type="cuda",enabled=device.type=="cuda"):
            parts.append(torch.sigmoid(model(batch.to(device,non_blocking=True))).float().cpu().numpy())
    return np.concatenate(parts)


def fit_epochs(x:np.ndarray,y:np.ndarray,w:np.ndarray,epochs:int,seed:int,
               device:torch.device)->TemporalCNN:
    seed_all(seed);model=TemporalCNN().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    positive=max(1,float(np.sum(y==1)));negative=max(1,float(np.sum(y==0)))
    pos_weight=torch.tensor(negative/positive,device=device)
    criterion=nn.BCEWithLogitsLoss(reduction="none",pos_weight=pos_weight)
    normalized=w/max(float(np.mean(w)),1e-8)
    for _ in range(epochs):
        model.train()
        for batch,target,weight in loader(x,y,normalized,True):
            batch=batch.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
            weight=weight.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda",enabled=device.type=="cuda"):
                loss=(criterion(model(batch),target)*weight).mean()
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),3.0);scaler.step(optimizer);scaler.update()
    return model


def choose_epoch(train:pd.DataFrame,sequences:np.ndarray,seed:int,device:torch.device)->tuple[int,dict[str,Any]]:
    inner_train,inner_valid=v37.split_inner(train)
    train_x=sequences[inner_train.seq_index.to_numpy(int)]
    valid_x=sequences[inner_valid.seq_index.to_numpy(int)]
    mean,std=normalizer(train_x);train_x=transform(train_x,mean,std);valid_x=transform(valid_x,mean,std)
    seed_all(seed);model=TemporalCNN().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    y=inner_train.y.to_numpy(np.float32);w=inner_train.cluster_weight.to_numpy(np.float32)
    normalized=w/max(float(np.mean(w)),1e-8)
    pos_weight=torch.tensor(max(1,float(np.sum(y==0)))/max(1,float(np.sum(y==1))),device=device)
    criterion=nn.BCEWithLogitsLoss(reduction="none",pos_weight=pos_weight)
    best_score=-np.inf;best_epoch=1;best_state=None;stale=0;history=[]
    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        for batch,target,weight in loader(train_x,y,normalized,True):
            batch=batch.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
            weight=weight.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda",enabled=device.type=="cuda"):
                loss=(criterion(model(batch),target)*weight).mean()
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),3.0);scaler.step(optimizer);scaler.update()
        probability=predict(model,valid_x,device);truth=inner_valid.y.to_numpy(int)
        auc=float(roc_auc_score(truth,probability));ba=float(balanced_accuracy_score(truth,probability>=.5))
        score=auc+.25*ba;history.append({"epoch":epoch,"auc":auc,"balanced_accuracy":ba})
        if score>best_score+1e-4:
            best_score=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if epoch>=6 and stale>=PATIENCE:break
    del model;torch.cuda.empty_cache()
    return best_epoch,{"inner_train_n":len(inner_train),"inner_valid_n":len(inner_valid),
                       "best_epoch":best_epoch,"best_score":float(best_score),
                       "best_metrics":history[best_epoch-1],"epochs_run":len(history)}


def market_oof(frame:pd.DataFrame,sequences:np.ndarray,base:pd.Series,market:str,device:torch.device):
    outputs=[];audits=[]
    for train,valid,fold in v37.chronological_folds(frame):
        seed=v36.SEED+fold+(0 if market=="US" else 100)
        epochs,audit=choose_epoch(train,sequences,seed,device)
        train_x=sequences[train.seq_index.to_numpy(int)];valid_x=sequences[valid.seq_index.to_numpy(int)]
        mean,std=normalizer(train_x);train_x=transform(train_x,mean,std);valid_x=transform(valid_x,mean,std)
        model=fit_epochs(train_x,train.y.to_numpy(np.float32),
                         train.cluster_weight.to_numpy(np.float32),epochs,seed+1000,device)
        cnn_probability=predict(model,valid_x,device);del model;torch.cuda.empty_cache()
        base_probability=valid.event_id.map(base).to_numpy(float)
        probability=BLEND_WEIGHT*cnn_probability+(1-BLEND_WEIGHT)*base_probability
        base_rows=valid.event_id.map(lambda value:value).to_frame(name="event_id")
        # Independent V48 mask already embedded in the V54 cache.
        base_frame=pd.read_csv(BASE_PATH,compression="gzip",usecols=["event_id","high_conf","confidence_signal"])
        high_map=base_frame.set_index("event_id").high_conf.astype(bool)
        confidence_map=base_frame.set_index("event_id").confidence_signal.astype(float)
        output=valid[["event_id","event_group_id","event_time_utc","market","ticker","source_family",
                      "form","y","fwd_ret_30m","text_cluster_id","cluster_weight"]].copy()
        output["prob"]=probability;output["cnn_prob"]=cnn_probability
        output["high_conf"]=output.event_id.map(high_map).to_numpy(bool)
        output["confidence_signal"]=output.event_id.map(confidence_map).to_numpy(float)
        output["fold"]=fold;output["family"]="V55_TEMPORAL_CNN_BLEND"
        seen=set(train.ticker.astype(str));output["new_issuer"]=~output.ticker.astype(str).isin(seen)
        outputs.append(output);audit.update({"market":market,"fold":fold,"outer_train_n":len(train),
            "outer_valid_n":len(valid),"train_end":train.event_time_utc.max(),
            "valid_start":valid.event_time_utc.min(),"blend_weight":BLEND_WEIGHT})
        audits.append(audit);print(f"[V55 CNN] {market} fold={fold} epochs={epochs} valid={len(valid)}",flush=True)
    return pd.concat(outputs,ignore_index=True),audits


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("V55 requires CUDA GPU")
    device=torch.device("cuda:0");archive=np.load(SEQUENCE_PATH,allow_pickle=False)
    event_ids=archive["event_id"].astype(str);sequences=archive["sequence"].astype(np.float32)
    index=pd.Series(np.arange(len(event_ids)),index=event_ids)
    data=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",low_memory=False,
                     dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    data["seq_index"]=data.event_id.map(index)
    if data.seq_index.isna().any() or len(sequences)!=len(data):raise RuntimeError("V55 sequence alignment")
    base_frame=pd.read_csv(BASE_PATH,compression="gzip",low_memory=False)
    base_probability=base_frame.set_index("event_id").prob.astype(float)
    outputs=[];audits=[]
    for market in ("US","KR"):
        result,market_audits=market_oof(data[data.market.eq(market)].copy(),sequences,
                                        base_probability,market,device)
        outputs.append(result);audits.extend(market_audits)
    selected=pd.concat(outputs,ignore_index=True);result=v44.summarize(selected)
    cnn=selected.copy();cnn["prob"]=cnn.cnn_prob;cnn_result=v44.summarize(cnn)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation="market-specific outer chronological OOF; epoch selected inner-only; full outer-train refit; fixed 50/50 blend"
    comparison={"version":"V55","hypothesis":"RAW_BAR_TEMPORAL_CNN",
        "dev_sha256":v37.sha256(DATA/"dev_contract_v36_labeled.csv.gz"),
        "sequence_sha256":v37.sha256(SEQUENCE_PATH),"base_sha256":v37.sha256(BASE_PATH),
        "device":torch.cuda.get_device_name(0),"models":{"V55_TEMPORAL_CNN_BLEND":result,
        "temporal_cnn_component":cnn_result},"fold_audits":audits,"validation":validation}
    robustness={"version":"V55","status":status,"hypothesis":"RAW_BAR_TEMPORAL_CNN",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "cnn_component":cnn_result,"fold_audits":audits,"seal_authorized":False}
    transfer={"version":"V55","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v55_temporal_cnn_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],
                      "gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
