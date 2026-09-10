"""Outcome-blind multilingual article embedding cache for historical news."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import runtime_limits  # configures full CPU affinity and exposes GPU 0

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data/historical_news_backfill/ARTICLES_NORMALIZED.parquet"
OUTPUT_DIR = ROOT / "data/historical_news_backfill/embeddings"
MODEL_ID = "intfloat/multilingual-e5-base"
SCHEMA_VERSION = "HIST_NEWS_MULTILINGUAL_EMBEDDING_V1"
LEDGER_NAME = "EMBEDDING_INDEX_LEDGER.jsonl"
INDEX_NAME = "EMBEDDING_INDEX.parquet"
STATE_NAME = "EMBEDDING_WORKER_STATE.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean_text(headline: Any, body: Any) -> str:
    """Create label-free E5 passage text and remove common wire attribution."""
    head = re.sub(r"\s+", " ", str(headline or "")).strip()
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    text = re.sub(
        r"^(?:\[[^\]]{1,60}\]|\([^)]{1,60}\)|(?:by\s+)?(?:reuters|associated press|ap|연합뉴스|뉴스1))\s*[-—:–]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    combined = f"{head}. {text}".strip(" .")
    return "passage: " + combined


def text_sha256(headline: Any, body: Any) -> str:
    return sha256_bytes(clean_text(headline, body).encode("utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for row in rows:
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def latest_index(ledger: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(ledger):
        uid = str(row.get("article_uid") or "")
        if uid:
            latest[uid] = row
    return latest


def resolve_cached_model() -> tuple[Path, str]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=MODEL_ID, local_files_only=True)).resolve()
    if not snapshot.is_dir():
        raise RuntimeError("LOCAL_EMBEDDING_MODEL_MISSING")
    revision = snapshot.name
    return snapshot, revision


def load_encoder(device: str) -> tuple[Callable[[list[str], int], np.ndarray], dict[str, Any]]:
    import torch
    from sentence_transformers import SentenceTransformer

    snapshot, revision = resolve_cached_model()
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    model = SentenceTransformer(str(snapshot), device=selected, local_files_only=True)

    def encode(texts: list[str], batch_size: int) -> np.ndarray:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    meta = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "model_snapshot": str(snapshot),
        "device": selected,
        "gpu_used": selected.startswith("cuda"),
        "target_or_outcome_training": False,
        "local_files_only": True,
    }
    return encode, meta


def write_shard(output_dir: Path, vectors: np.ndarray, identities: list[tuple[str, str]]) -> tuple[Path, str]:
    vectors16 = np.asarray(vectors, dtype=np.float16)
    semantic = hashlib.sha256()
    semantic.update(vectors16.tobytes(order="C"))
    for uid, digest in identities:
        semantic.update(uid.encode("utf-8"))
        semantic.update(digest.encode("ascii"))
    shard_hash = semantic.hexdigest()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"EMB_{shard_hash}.npy"
    if not path.is_file():
        temp = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temp.open("wb") as handle:
            np.save(handle, vectors16, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    observed = sha256_bytes(path.read_bytes())
    return path, observed


def write_index_snapshot(output_dir: Path, latest: dict[str, dict[str, Any]]) -> Path:
    columns = [
        "schema_version", "article_uid", "text_sha256", "model_id", "model_revision",
        "dimension", "dtype", "normalized", "shard_path", "shard_index",
        "shard_sha256", "embedded_at_utc", "gpu_used",
    ]
    frame = pd.DataFrame(latest.values())
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns].sort_values("article_uid", kind="mergesort").reset_index(drop=True)
    path = output_dir / INDEX_NAME
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)
    return path


def run_once(
    *,
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
    max_items: int,
    batch_size: int,
    device: str,
    encoder: Callable[[list[str], int], np.ndarray] | None = None,
    encoder_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise RuntimeError(f"ARTICLE_INPUT_MISSING:{input_path}")
    articles = pd.read_parquet(input_path, columns=["article_uid", "headline", "body"])
    articles["article_uid"] = articles["article_uid"].fillna("").astype(str)
    if articles["article_uid"].eq("").any() or articles["article_uid"].duplicated().any():
        raise RuntimeError("ARTICLE_UID_MISSING_OR_DUPLICATE")
    articles["text_sha256"] = [text_sha256(h, b) for h, b in zip(articles["headline"], articles["body"])]
    ledger = output_dir / LEDGER_NAME
    latest = latest_index(ledger)
    pending = articles.loc[
        [
            latest.get(uid, {}).get("text_sha256") != digest
            for uid, digest in zip(articles["article_uid"], articles["text_sha256"])
        ]
    ].head(max_items)
    backlog_before = int(sum(
        latest.get(uid, {}).get("text_sha256") != digest
        for uid, digest in zip(articles["article_uid"], articles["text_sha256"])
    ))
    if len(pending) == 0:
        state = {
            "schema_version": SCHEMA_VERSION,
            "status": "NO_PROGRESS",
            "articles_total": int(len(articles)),
            "embeddings_current": int(len(latest)),
            "embedding_backlog": 0,
            "gpu_used": bool((encoder_meta or {}).get("gpu_used", False)),
            "outcome_fields_read": [],
            "updated_at_utc": now_utc(),
        }
        atomic_json(output_dir / STATE_NAME, state)
        return {**state, "items_processed": 0, "new_embeddings": 0}
    if encoder is None:
        encoder, encoder_meta = load_encoder(device)
    encoder_meta = dict(encoder_meta or {})
    texts = [clean_text(h, b) for h, b in zip(pending["headline"], pending["body"])]
    vectors = np.asarray(encoder(texts, batch_size), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(pending) or not np.isfinite(vectors).all():
        raise RuntimeError(f"INVALID_EMBEDDING_MATRIX:{vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("ZERO_NORM_EMBEDDING")
    vectors = vectors / norms[:, None]
    identities = list(zip(pending["article_uid"].tolist(), pending["text_sha256"].tolist()))
    shard_path, shard_sha = write_shard(output_dir, vectors, identities)
    timestamp = now_utc()
    rows: list[dict[str, Any]] = []
    for index, (uid, digest) in enumerate(identities):
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "article_uid": uid,
            "text_sha256": digest,
            "model_id": str(encoder_meta.get("model_id", MODEL_ID)),
            "model_revision": str(encoder_meta.get("model_revision", "TEST_OR_UNKNOWN")),
            "dimension": int(vectors.shape[1]),
            "dtype": "float16",
            "normalized": True,
            "shard_path": shard_path.relative_to(output_dir).as_posix(),
            "shard_index": index,
            "shard_sha256": shard_sha,
            "embedded_at_utc": timestamp,
            "gpu_used": bool(encoder_meta.get("gpu_used", False)),
        })
    append_jsonl(ledger, rows)
    for row in rows:
        latest[row["article_uid"]] = row
    index_path = write_index_snapshot(output_dir, latest)
    meta = {
        "schema_version": SCHEMA_VERSION,
        **encoder_meta,
        "dimension": int(vectors.shape[1]),
        "dtype": "float16",
        "normalized": True,
        "text_recipe": "E5 passage prefix; headline + body; common publisher attribution stripped",
        "outcome_fields_read": [],
        "target_or_outcome_training": False,
    }
    atomic_json(output_dir / "EMBEDDING_MODEL_FREEZE.json", meta)
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "articles_total": int(len(articles)),
        "items_processed": int(len(pending)),
        "new_embeddings": int(len(rows)),
        "embeddings_current": int(len(latest)),
        "embedding_backlog": int(max(0, backlog_before - len(rows))),
        "gpu_used": bool(encoder_meta.get("gpu_used", False)),
        "device": encoder_meta.get("device"),
        "model_id": encoder_meta.get("model_id", MODEL_ID),
        "model_revision": encoder_meta.get("model_revision"),
        "index_path": str(index_path),
        "outcome_fields_read": [],
        "updated_at_utc": now_utc(),
    }
    atomic_json(output_dir / STATE_NAME, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.once:
        parser.error("--once is required")
    if args.max_items < 0 or args.batch_size < 1:
        parser.error("invalid batch limits")
    try:
        result = run_once(
            input_path=args.input_path,
            output_dir=args.output_dir,
            max_items=args.max_items,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as error:
        print(json.dumps({"status": "FATAL", "error_class": type(error).__name__, "message": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
