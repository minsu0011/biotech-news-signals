"""V57: supervised, fold-specific language adaptation for the US SEC DGP."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

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
from transformers import AutoModel,AutoTokenizer

import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT",str(ROOT/"output_V57")))
CACHE=ROOT/"cache"
BASE_PATH=CACHE/"v56_semantic_retrieval_oof.csv.gz"
MODEL_NAME="intfloat/multilingual-e5-base"
MAX_LENGTH=256
BATCH_SIZE=16
MAX_EPOCHS=5
BLEND_WEIGHT=.50


class SECClassifier(nn.Module):
    def __init__(self)->None:
        super().__init__();self.encoder=AutoModel.from_pretrained(MODEL_NAME)
        for parameter in self.encoder.parameters():parameter.requires_grad=False
        for layer in self.encoder.encoder.layer[-2:]:
            for parameter in layer.parameters():parameter.requires_grad=True
        self.head=nn.Sequential(nn.Dropout(.15),nn.Linear(self.encoder.config.hidden_size,1))
    def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor)->torch.Tensor:
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        mask=attention_mask.unsqueeze(-1);pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.head(pooled).squeeze(1)


def seed_all(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def batches(ids:np.ndarray,masks:np.ndarray,y:np.ndarray,w:np.ndarray,
            indices:np.ndarray,shuffle:bool)->DataLoader:
    dataset=TensorDataset(torch.from_numpy(ids[indices].astype(np.int64)),
        torch.from_numpy(masks[indices].astype(np.int64)),torch.from_numpy(y[indices].astype(np.float32)),
        torch.from_numpy(w[indices].astype(np.float32)))
    return DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=0,
                      pin_memory=True,drop_last=False)


@torch.inference_mode()
def predict(model:nn.Module,ids:np.ndarray,masks:np.ndarray,indices:np.ndarray,
            device:torch.device)->np.ndarray:
    model.eval();parts=[];dummy=np.zeros(len(ids),np.float32);one=np.ones(len(ids),np.float32)
    for input_ids,attention,_,_ in batches(ids,masks,dummy,one,indices,False):
        with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
            logits=model(input_ids.to(device,non_blocking=True),attention.to(device,non_blocking=True))
        parts.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(parts)


def train_epoch(model:nn.Module,optimizer:torch.optim.Optimizer,scaler:torch.amp.GradScaler,
                ids:np.ndarray,masks:np.ndarray,y:np.ndarray,w:np.ndarray,indices:np.ndarray,
                device:torch.device)->None:
    target=y[indices];positive=max(1,float(np.sum(target==1)));negative=max(1,float(np.sum(target==0)))
    criterion=nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.tensor(negative/positive,device=device))
    normalized=w/max(float(np.mean(w[indices])),1e-8);model.train()
    for input_ids,attention,label,weight in batches(ids,masks,y,normalized,indices,True):
        input_ids=input_ids.to(device,non_blocking=True);attention=attention.to(device,non_blocking=True)
        label=label.to(device,non_blocking=True);weight=weight.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
            loss=(criterion(model(input_ids,attention),label)*weight).mean()
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
        scaler.step(optimizer);scaler.update()


def new_model(seed:int,device:torch.device)->tuple[SECClassifier,torch.optim.Optimizer,torch.amp.GradScaler]:
    seed_all(seed);model=SECClassifier().to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-5,weight_decay=.01)
    return model,optimizer,torch.amp.GradScaler("cuda")


def select_epoch(inner_train:pd.DataFrame,inner_valid:pd.DataFrame,ids:np.ndarray,masks:np.ndarray,
                 y:np.ndarray,w:np.ndarray,seed:int,device:torch.device)->tuple[int,dict[str,Any]]:
    model,optimizer,scaler=new_model(seed,device);best=-np.inf;best_epoch=1;history=[]
    train_index=inner_train.token_index.to_numpy(int);valid_index=inner_valid.token_index.to_numpy(int)
    for epoch in range(1,MAX_EPOCHS+1):
        train_epoch(model,optimizer,scaler,ids,masks,y,w,train_index,device)
        probability=predict(model,ids,masks,valid_index,device);truth=y[valid_index].astype(int)
        auc=float(roc_auc_score(truth,probability));ba=float(balanced_accuracy_score(truth,probability>=.5))
        score=auc+.25*ba;history.append({"epoch":epoch,"auc":auc,"balanced_accuracy":ba})
        if score>best:best=score;best_epoch=epoch
    del model;torch.cuda.empty_cache()
    return best_epoch,{"inner_train_n":len(inner_train),"inner_valid_n":len(inner_valid),
        "best_epoch":best_epoch,"best_score":float(best),"best_metrics":history[best_epoch-1],
        "epochs_run":len(history)}


def main()->None:
    runtime_limits.configure();OUT.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("V57 requires CUDA")
    device=torch.device("cuda:0")
    data=pd.read_csv(DATA/"dev_contract_v36_labeled.csv.gz",compression="gzip",low_memory=False,
                     dtype={"ticker":str},parse_dates=["event_time_utc"])
    events=pd.read_csv(DATA/"dev_contract_v36_events.csv.gz",compression="gzip",low_memory=False,
                       usecols=["event_id","url","article_id"])
    data=data.merge(events,on="event_id",how="left",validate="one_to_one")
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    data["text_cluster_id"]=data.apply(v37.normalized_cluster,axis=1)
    data["cluster_weight"]=1./data.groupby("text_cluster_id").event_id.transform("size")
    data["semantic_text"]=[v37.semantic_segment(row) for _,row in data.iterrows()]
    sec=data[data.source_family.eq("US_SEC")].copy().reset_index(drop=True)
    sec["token_index"]=np.arange(len(sec));tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded=tokenizer(("passage: "+sec.semantic_text).tolist(),padding="max_length",truncation=True,
                      max_length=MAX_LENGTH,return_tensors="np")
    input_ids=encoded["input_ids"].astype(np.int32);attention=encoded["attention_mask"].astype(np.int8)
    y=sec.y.to_numpy(np.float32);weight=sec.cluster_weight.to_numpy(np.float32)
    base=pd.read_csv(BASE_PATH,compression="gzip",low_memory=False);base_index=base.set_index("event_id")
    outputs=[];audits=[]
    # The outer blocks remain the canonical market blocks.  Only their SEC rows
    # are used for language fitting and prediction; all other DGPs retain V56.
    us=data[data.market.eq("US")].copy();sec_index=sec.set_index("event_id").token_index
    for train,valid,fold in v37.chronological_folds(us):
        train_sec=train[train.source_family.eq("US_SEC")].copy();valid_sec=valid[valid.source_family.eq("US_SEC")].copy()
        train_sec["token_index"]=train_sec.event_id.map(sec_index);valid_sec["token_index"]=valid_sec.event_id.map(sec_index)
        inner_train,inner_valid=v37.split_inner(train_sec)
        seed=v36.SEED+200+fold;epochs,audit=select_epoch(inner_train,inner_valid,input_ids,attention,y,
                                                         weight,seed,device)
        model,optimizer,scaler=new_model(seed+1000,device);train_indices=train_sec.token_index.to_numpy(int)
        for _ in range(epochs):train_epoch(model,optimizer,scaler,input_ids,attention,y,weight,train_indices,device)
        transformer_probability=predict(model,input_ids,attention,valid_sec.token_index.to_numpy(int),device)
        del model;torch.cuda.empty_cache()
        probability=valid.event_id.map(base_index.prob).to_numpy(float);route=valid.source_family.eq("US_SEC").to_numpy()
        probability[route]=(BLEND_WEIGHT*transformer_probability+(1-BLEND_WEIGHT)*probability[route])
        transformer_full=np.full(len(valid),np.nan);transformer_full[route]=transformer_probability
        output=valid[["event_id","event_group_id","event_time_utc","market","ticker","source_family",
                      "form","y","fwd_ret_30m","text_cluster_id","cluster_weight"]].copy()
        output["prob"]=probability;output["transformer_prob"]=transformer_full
        output["high_conf"]=output.event_id.map(base_index.high_conf).to_numpy(bool)
        output["confidence_signal"]=output.event_id.map(base_index.confidence_signal).to_numpy(float)
        output["fold"]=fold;output["family"]="V57_SEC_TRANSFORMER_BLEND"
        seen=set(train.ticker.astype(str));output["new_issuer"]=~output.ticker.astype(str).isin(seen)
        outputs.append(output);audit.update({"market":"US","fold":fold,"outer_train_sec_n":len(train_sec),
            "outer_valid_sec_n":len(valid_sec),"outer_valid_n":len(valid),"blend_weight":BLEND_WEIGHT,
            "train_end":train.event_time_utc.max(),"valid_start":valid.event_time_utc.min()})
        audits.append(audit);print(f"[V57 SEC] fold={fold} epochs={epochs} train={len(train_sec)} valid={len(valid_sec)}",flush=True)
    # KR is unchanged and its OOF metadata is already causally valid in V56.
    outputs.append(base[base.market.eq("KR")].copy())
    selected=pd.concat(outputs,ignore_index=True);result=v44.summarize(selected)
    status="ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation=("canonical US outer folds; epoch selected on inner-past SEC only; full outer SEC refit; "
                "last two encoder layers adapted; fixed 50/50 routing; KR and non-SEC unchanged")
    comparison={"version":"V57","hypothesis":"SUPERVISED_SEC_LANGUAGE_ADAPTATION",
        "dev_sha256":v37.sha256(DATA/"dev_contract_v36_labeled.csv.gz"),"base_sha256":v37.sha256(BASE_PATH),
        "model":MODEL_NAME,"max_length":MAX_LENGTH,"adapted_encoder_layers":2,
        "device":torch.cuda.get_device_name(0),"models":{"V57_SEC_TRANSFORMER_BLEND":result},
        "fold_audits":audits,"validation":validation}
    robustness={"version":"V57","status":status,"hypothesis":"SUPERVISED_SEC_LANGUAGE_ADAPTATION",
        "opened_dev_only":True,"v36plus_seal_outcomes_loaded":False,"selected":result,
        "fold_audits":audits,"seal_authorized":False}
    transfer={"version":"V57","seal_outcomes_loaded":False,
        "selected_oof_by_source":result["metrics"]["by_source_family"],
        "selected_oof_by_market":result["metrics"]["by_market"],
        "selected_oof_new_issuer":result["metrics"]["new_issuer"],"method":validation}
    v37.atomic_json(comparison,OUT/"MODEL_COMPARISON.json")
    v37.atomic_json(robustness,OUT/"DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(transfer,OUT/"SOURCE_TRANSFER_REPORT.json")
    selected.to_csv(CACHE/"v57_sec_transformer_oof.csv.gz",index=False,compression="gzip")
    print(json.dumps({"status":status,"metrics":result["metrics"],
                      "gate":result["research_gate"]},indent=2,default=str))


if __name__=="__main__":main()
