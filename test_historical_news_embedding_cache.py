from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import historical_news_embedding_cache as module


def fake_encoder(texts: list[str], _batch_size: int) -> np.ndarray:
    rows = []
    for text in texts:
        base = float(len(text))
        rows.append([base + 1.0, base + 2.0, base + 3.0, base + 4.0])
    return np.asarray(rows, dtype=np.float32)


def test_clean_text_is_outcome_blind_and_strips_wire_prefix() -> None:
    value = module.clean_text("Headline", "(Reuters) - Body text")
    assert value == "passage: Headline. Body text"
    assert "label" not in value.lower()


def test_resume_embeds_only_missing_or_changed_text(tmp_path: Path) -> None:
    source = tmp_path / "articles.parquet"
    output = tmp_path / "embeddings"
    pd.DataFrame(
        {
            "article_uid": ["a", "b", "c"],
            "headline": ["one", "two", "three"],
            "body": ["alpha", "beta", "gamma"],
            "y": [1, 0, 1],  # must never be requested from parquet
        }
    ).to_parquet(source, index=False)
    meta = {"model_id": "fake", "model_revision": "r1", "device": "cpu", "gpu_used": False}
    first = module.run_once(
        input_path=source, output_dir=output, max_items=2, batch_size=2,
        device="cpu", encoder=fake_encoder, encoder_meta=meta,
    )
    assert first["new_embeddings"] == 2
    second = module.run_once(
        input_path=source, output_dir=output, max_items=10, batch_size=2,
        device="cpu", encoder=fake_encoder, encoder_meta=meta,
    )
    assert second["new_embeddings"] == 1
    third = module.run_once(
        input_path=source, output_dir=output, max_items=10, batch_size=2,
        device="cpu", encoder=fake_encoder, encoder_meta=meta,
    )
    assert third["status"] == "NO_PROGRESS"
    assert len(pd.read_parquet(output / module.INDEX_NAME)) == 3


def test_shard_vectors_are_float16_and_normalized(tmp_path: Path) -> None:
    source = tmp_path / "articles.parquet"
    output = tmp_path / "embeddings"
    pd.DataFrame({"article_uid": ["a"], "headline": ["x"], "body": ["y"]}).to_parquet(source, index=False)
    module.run_once(
        input_path=source, output_dir=output, max_items=1, batch_size=1,
        device="cpu", encoder=fake_encoder,
        encoder_meta={"model_id": "fake", "model_revision": "r1", "device": "cpu", "gpu_used": False},
    )
    index = pd.read_parquet(output / module.INDEX_NAME)
    vector = np.load(output / index.iloc[0]["shard_path"], allow_pickle=False)[0]
    assert vector.dtype == np.float16
    assert abs(float(np.linalg.norm(vector.astype(np.float32))) - 1.0) < 0.002
