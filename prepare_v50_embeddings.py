"""Create outcome-free frozen multilingual E5 text embeddings on RTX 5080."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import experiment_v37_text as v37


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
CACHE=ROOT/"cache"
MODEL_NAME="intfloat/multilingual-e5-base"


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main()->None:
    if not torch.cuda.is_available():raise RuntimeError("V50 requires CUDA PyTorch")
    torch.manual_seed(20260827);torch.cuda.manual_seed_all(20260827)
    path=DATA/"dev_contract_v36_labeled.csv.gz"
    data=pd.read_csv(path,compression="gzip",low_memory=False,
        usecols=["event_id","source_family","form","event_type","headline","body"])
    data["headline"]=data.headline.fillna("").astype(str);data["body"]=data.body.fillna("").astype(str)
    headline=[f"passage: source {row.source_family} form {row.form} event {row.event_type} {row.headline}"
              for row in data.itertuples(index=False)]
    semantic=["passage: "+v37.semantic_segment(row) for _,row in data.iterrows()]
    model=SentenceTransformer(MODEL_NAME,device="cuda");model.max_seq_length=384
    headline_embedding=model.encode(headline,batch_size=64,show_progress_bar=True,
        convert_to_numpy=True,normalize_embeddings=True,precision="float32")
    semantic_embedding=model.encode(semantic,batch_size=32,show_progress_bar=True,
        convert_to_numpy=True,normalize_embeddings=True,precision="float32")
    if not np.isfinite(headline_embedding).all() or not np.isfinite(semantic_embedding).all():
        raise RuntimeError("V50 non-finite embedding")
    output=CACHE/"v50_multilingual_e5_embeddings.npz"
    np.savez_compressed(output,event_id=data.event_id.astype(str).to_numpy(),
                        headline=headline_embedding.astype(np.float16),
                        semantic=semantic_embedding.astype(np.float16))
    report={"model":MODEL_NAME,"model_revision":"resolved_by_huggingface_cache",
        "outcomes_loaded":False,"rows":len(data),"dimensions":int(headline_embedding.shape[1]),
        "headline_max_tokens":384,"semantic_max_tokens":384,"normalized":True,
        "device":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,
        "dev_text_input_sha256":v37.sha256(path),"embedding_path":str(output.relative_to(ROOT)),
        "embedding_sha256":sha256(output),"openmp_workaround_scope":"embedding process only",
        "headline_repeat_max_abs_error":float(np.max(np.abs(headline_embedding[:16]-model.encode(
            headline[:16],batch_size=16,convert_to_numpy=True,normalize_embeddings=True)) ))}
    (CACHE/"V50_EMBEDDING_STATUS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
