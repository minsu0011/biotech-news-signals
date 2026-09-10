"""V82 GPU masked-state self-supervised direction challenger.

For every embargoed inner/outer past-training cut, a tabular encoder first
learns without direction or return labels.  Two independently masked views of
causal pre-event numeric state and source/event/form categories are reconstructed
and their latent states are constrained to agree.  The encoder is then frozen;
only a linear direction head is fitted with training-cut labels.  Validation or
outer rows never participate in preprocessing, pretraining, or head fitting.

This is distinct from supervised CatBoost/MLP/CNN models and from frozen text
embeddings: the material information is a cut-local self-supervised causal-state
representation.  Inner-past OOF chooses exact V69 or one of three fixed blends.
V69 confidence/high_conf remain frozen.  Authority is immutable V36 DEV plus
atomic output_V69.  Material failure restores the exact complete V69 frame and
the controller recomputes the canonical 14 gates.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import gzip
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import autonomous_v37plus as controller
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
V69_COMMIT_PATH = V69_DIR / "COMMIT.json"
V69_MANIFEST_PATH = V69_DIR / "ARTIFACT_MANIFEST.json"
V69_ROBUSTNESS_PATH = V69_DIR / "DEV_ROBUSTNESS_REPORT.json"
OUT = Path(
    os.environ.get(
        "MARKET_BIO_VERSION_OUTPUT",
        str(ROOT / "staging" / "V82_MASKED_STATE_SELF_SUPERVISED_GPU_V1"),
    )
)

VERSION = 82
HYPOTHESIS = "MASKED_CAUSAL_STATE_SELF_SUPERVISED_GPU_V1"
EXPECTED_SHA256 = {
    DEV_PATH: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V69_PATH: "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002",
}
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V69_COMMIT_SHA256 = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA256 = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EXPECTED_V69_ROBUSTNESS_SHA256 = "778c5a87e4be328d702f85eb7298bcbb70e8b54d3429b19a0dc6e2c7e156cce1"
EXPECTED_DEV_ROWS = 9462
EXPECTED_OOF_ROWS = 7568

EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
MASK_RATE = 0.25
EMBEDDING_DIM = 8
LATENT_DIM = 32
PRETRAIN_EPOCHS = 18
HEAD_EPOCHS = 35
BATCH_SIZE = 256
PRETRAIN_LR = 1.0e-3
HEAD_LR = 2.0e-3
CONSISTENCY_WEIGHT = 0.10
CATEGORICAL_LOSS_WEIGHT = 0.25
SEED = 8201
BOOTSTRAP_DRAWS = 2000

NUMERIC_COLUMNS = (
    "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
    "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
    "pre_ret_90", "pre_ret_120", "pre_vol_5", "pre_vol_10", "pre_vol_30",
    "pre_vol_60", "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30",
    "volume_ratio_10_60", "range_5m", "range_10m", "range_30m", "range_60m",
    "close_position_10", "close_position_30", "close_position_60", "entry_bar_ret",
    "entry_bar_range", "intraday_ret_open", "return_autocorr_30", "up_fraction_10",
    "up_fraction_30", "trend_slope_30", "trend_slope_60", "vwap_distance_30",
    "log_entry_price", "minutes_from_open", "minutes_to_close", "event_positive_kw",
    "event_negative_kw", "event_financing_kw", "event_trial_kw", "event_regulatory_kw",
    "event_ma_kw", "event_earnings_kw", "headline_len", "body_len", "benchmark_ret_2",
    "benchmark_ret_5", "benchmark_ret_15", "benchmark_ret_30", "benchmark_ret_60",
)
CATEGORICAL_COLUMNS = ("source_family", "event_type", "form")

ARCHITECTURES = (
    {"name": "MASKED_STATE_LOGIT_W25", "mode": "fixed", "weight": 0.25},
    {"name": "MASKED_STATE_LOGIT_W50", "mode": "fixed", "weight": 0.50},
    {"name": "RECONSTRUCTION_RELIABILITY_W50", "mode": "reliability", "weight": 0.50},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    raw = (json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(raw, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.name.endswith(".gz"):
        with temporary.open("wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
                compressed.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary.write_bytes(raw)
    os.replace(temporary, path)


def metric(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int | None]:
    y = frame.y.to_numpy(int)
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    return {
        "n": len(frame),
        "auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(prediction == y)),
        "pred_up": float(prediction.mean()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def verify_and_load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"required V82 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V82 input changed: {path.name}")
        hashes[str(path.relative_to(ROOT))] = actual
    require(sha256(V69_COMMIT_PATH) == EXPECTED_V69_COMMIT_SHA256, "V69 COMMIT changed")
    require(sha256(V69_MANIFEST_PATH) == EXPECTED_V69_MANIFEST_SHA256, "V69 manifest changed")
    require(sha256(V69_ROBUSTNESS_PATH) == EXPECTED_V69_ROBUSTNESS_SHA256, "V69 robustness changed")
    commit = json.loads(V69_COMMIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(V69_MANIFEST_PATH.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment mismatch")
    require(commit.get("manifest_sha256") == EXPECTED_V69_MANIFEST_SHA256, "V69 COMMIT binding mismatch")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest experiment mismatch")
    files = manifest.get("files", {})
    require(files.get(V69_PATH.name, {}).get("sha256") == EXPECTED_SHA256[V69_PATH], "V69 OOF unbound")
    require(
        files.get(V69_ROBUSTNESS_PATH.name, {}).get("sha256") == EXPECTED_V69_ROBUSTNESS_SHA256,
        "V69 robustness unbound",
    )
    require(torch.cuda.is_available(), "V82 requires CUDA")
    data = pd.read_csv(DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    champion = pd.read_csv(V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    require(len(data) == EXPECTED_DEV_ROWS and data.event_id.is_unique, "immutable DEV changed")
    require(len(champion) == EXPECTED_OOF_ROWS and champion.event_id.is_unique, "V69 OOF changed")
    require(set(champion.event_id).issubset(set(data.event_id)), "V69 event outside immutable DEV")
    missing = [column for column in (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS) if column not in data]
    require(not missing, f"immutable DEV missing self-supervised features: {missing}")
    data["event_time_utc"] = pd.to_datetime(data.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    audit = {
        "input_sha256": hashes,
        "authorized_data_inputs": [str(DEV_PATH.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT))],
        "authorized_v69_integrity_inputs": [
            str(V69_COMMIT_PATH.relative_to(ROOT)),
            str(V69_MANIFEST_PATH.relative_to(ROOT)),
            str(V69_ROBUSTNESS_PATH.relative_to(ROOT)),
        ],
        "immutable_dev_only": True,
        "extension_data_opened": False,
        "assignment_file_opened": False,
        "reserved_evaluation_opened": False,
        "failed_v71_v81_results_opened": False,
        "dev_rows": len(data),
        "v69_oof_rows": len(champion),
        "v69_atomic_commit_verified": True,
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "pretraining_uses_direction_or_return_label": False,
    }
    return data, champion, audit


class CutPreprocessor:
    def fit(self, frame: pd.DataFrame) -> "CutPreprocessor":
        numeric = frame[list(NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        numeric[~np.isfinite(numeric)] = np.nan
        self.median = np.nanmedian(numeric, axis=0)
        self.median[~np.isfinite(self.median)] = 0.0
        filled = np.where(np.isfinite(numeric), numeric, self.median)
        self.scale = np.nanstd(filled, axis=0)
        self.scale[~np.isfinite(self.scale) | (self.scale < 1e-8)] = 1.0
        self.category_maps: list[dict[str, int]] = []
        self.category_sizes: list[int] = []
        for column in CATEGORICAL_COLUMNS:
            values = sorted(frame[column].fillna("").astype(str).unique())
            mapping = {value: index + 2 for index, value in enumerate(values)}
            self.category_maps.append(mapping)
            self.category_sizes.append(len(mapping) + 2)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = frame[list(NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.median)
        numeric = np.clip((filled - self.median) / self.scale, -8.0, 8.0).astype(np.float32)
        categories = np.empty((len(frame), len(CATEGORICAL_COLUMNS)), dtype=np.int64)
        for position, (column, mapping) in enumerate(zip(CATEGORICAL_COLUMNS, self.category_maps)):
            categories[:, position] = [mapping.get(value, 0) for value in frame[column].fillna("").astype(str)]
        return numeric, observed.astype(np.float32), categories


class MaskedStateEncoder(nn.Module):
    def __init__(self, numeric_n: int, category_sizes: list[int]):
        super().__init__()
        self.numeric_n = numeric_n
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, EMBEDDING_DIM) for size in category_sizes]
        )
        input_n = 3 * numeric_n + EMBEDDING_DIM * len(category_sizes)
        self.encoder = nn.Sequential(
            nn.Linear(input_n, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
        )
        self.numeric_decoder = nn.Linear(LATENT_DIM, numeric_n)
        self.category_decoders = nn.ModuleList(
            [nn.Linear(LATENT_DIM, size) for size in category_sizes]
        )

    def encode(
        self,
        numeric: torch.Tensor,
        observed: torch.Tensor,
        categories: torch.Tensor,
        numeric_mask: torch.Tensor,
        category_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_numeric = numeric.masked_fill(numeric_mask, 0.0)
        category_parts = []
        for position, embedding in enumerate(self.embeddings):
            values = categories[:, position].clone()
            values[category_mask[:, position]] = 1
            category_parts.append(embedding(values))
        inputs = torch.cat(
            [masked_numeric, observed, numeric_mask.float(), *category_parts], dim=1
        )
        return self.encoder(inputs)

    def reconstruct(self, latent: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.numeric_decoder(latent), [head(latent) for head in self.category_decoders]


def tensor_loader(
    numeric: np.ndarray,
    observed: np.ndarray,
    categories: np.ndarray,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(numeric),
        torch.from_numpy(observed),
        torch.from_numpy(categories),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


def random_masks(
    observed: torch.Tensor,
    categories: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    numeric_mask = (torch.rand_like(observed) < MASK_RATE) & observed.bool()
    category_mask = torch.rand(categories.shape, device=categories.device) < MASK_RATE
    return numeric_mask, category_mask


def pretrain_encoder(
    model: MaskedStateEncoder,
    numeric: np.ndarray,
    observed: np.ndarray,
    categories: np.ndarray,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    loader = tensor_loader(numeric, observed, categories, True, seed)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(PRETRAIN_EPOCHS):
        totals = np.zeros(4, dtype=float)
        batches = 0
        for values, present, cats in loader:
            values = values.to(device, non_blocking=True)
            present = present.to(device, non_blocking=True)
            cats = cats.to(device, non_blocking=True)
            mask_num_1, mask_cat_1 = random_masks(present, cats)
            mask_num_2, mask_cat_2 = random_masks(present, cats)
            latent_1 = model.encode(values, present, cats, mask_num_1, mask_cat_1)
            latent_2 = model.encode(values, present, cats, mask_num_2, mask_cat_2)
            reconstructed_numeric, reconstructed_categories = model.reconstruct(latent_1)
            if mask_num_1.any():
                numeric_loss = torch.mean((reconstructed_numeric[mask_num_1] - values[mask_num_1]) ** 2)
            else:
                numeric_loss = torch.zeros((), device=device)
            category_losses = []
            for position, logits in enumerate(reconstructed_categories):
                selected = mask_cat_1[:, position]
                if selected.any():
                    category_losses.append(nn.functional.cross_entropy(logits[selected], cats[selected, position]))
            categorical_loss = (
                torch.stack(category_losses).mean()
                if category_losses
                else torch.zeros((), device=device)
            )
            consistency_loss = torch.mean(
                1.0 - nn.functional.cosine_similarity(latent_1, latent_2, dim=1)
            )
            loss = numeric_loss + CATEGORICAL_LOSS_WEIGHT * categorical_loss + CONSISTENCY_WEIGHT * consistency_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            totals += [float(loss.detach()), float(numeric_loss.detach()),
                       float(categorical_loss.detach()), float(consistency_loss.detach())]
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": totals[0] / batches,
                "numeric_reconstruction_loss": totals[1] / batches,
                "categorical_reconstruction_loss": totals[2] / batches,
                "dual_mask_consistency_loss": totals[3] / batches,
            }
        )
    return history


@torch.inference_mode()
def representations_and_error(
    model: MaskedStateEncoder,
    numeric: np.ndarray,
    observed: np.ndarray,
    categories: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    latent_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    for values, present, cats in tensor_loader(numeric, observed, categories, False, SEED):
        values = values.to(device, non_blocking=True)
        present = present.to(device, non_blocking=True)
        cats = cats.to(device, non_blocking=True)
        empty_num = torch.zeros_like(present, dtype=torch.bool)
        empty_cat = torch.zeros_like(cats, dtype=torch.bool)
        latent = model.encode(values, present, cats, empty_num, empty_cat)
        reconstructed, _ = model.reconstruct(latent)
        squared = (reconstructed - values) ** 2 * present
        denominator = torch.clamp(present.sum(dim=1), min=1.0)
        error = squared.sum(dim=1) / denominator
        latent_parts.append(latent.cpu().numpy())
        error_parts.append(error.cpu().numpy())
    return np.concatenate(latent_parts), np.concatenate(error_parts)


def group_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(frame.event_group_id.astype(str)).event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= max(float(weights.mean()), 1e-12)
    return weights.astype(np.float32)


def fit_direction_head(
    latent: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    seed: int,
) -> nn.Linear:
    seed_everything(seed)
    head = nn.Linear(LATENT_DIM, 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=1e-3)
    x = torch.from_numpy(latent.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    w = torch.from_numpy(weights.astype(np.float32))
    dataset = TensorDataset(x, y, w)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        pin_memory=True, generator=generator)
    positives = max(1.0, float(np.sum(labels == 1)))
    negatives = max(1.0, float(np.sum(labels == 0)))
    positive_weight = torch.tensor(negatives / positives, device=device)
    for _ in range(HEAD_EPOCHS):
        head.train()
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_w = batch_w.to(device, non_blocking=True)
            logits = head(batch_x).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, batch_y, pos_weight=positive_weight, reduction="none"
            )
            loss = torch.mean(loss * batch_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head


@torch.inference_mode()
def predict_head(head: nn.Linear, latent: np.ndarray, device: torch.device) -> np.ndarray:
    head.eval()
    values = torch.from_numpy(latent.astype(np.float32))
    parts: list[np.ndarray] = []
    for (batch,) in DataLoader(TensorDataset(values), batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True):
        parts.append(torch.sigmoid(head(batch.to(device, non_blocking=True)).squeeze(1)).cpu().numpy())
    return np.concatenate(parts).astype(float)


def fit_self_supervised_direction(
    train: pd.DataFrame,
    target: pd.DataFrame,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    require(len(train) >= 300 and train.y.nunique() == 2, "self-supervised fit support insufficient")
    seed_everything(seed)
    processor = CutPreprocessor().fit(train)
    train_numeric, train_observed, train_categories = processor.transform(train)
    target_numeric, target_observed, target_categories = processor.transform(target)
    model = MaskedStateEncoder(len(NUMERIC_COLUMNS), processor.category_sizes).to(device)
    history = pretrain_encoder(
        model, train_numeric, train_observed, train_categories, device, seed
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    train_latent, train_error = representations_and_error(
        model, train_numeric, train_observed, train_categories, device
    )
    target_latent, target_error = representations_and_error(
        model, target_numeric, target_observed, target_categories, device
    )
    head = fit_direction_head(
        train_latent, train.y.to_numpy(int), group_weights(train), device, seed + 1000
    )
    probability = predict_head(head, target_latent, device)
    reference_error = max(float(np.median(train_error)), 1e-6)
    reliability = np.clip(2.0 / (1.0 + target_error / reference_error), 0.10, 1.0)
    require(np.isfinite(probability).all() and np.isfinite(reliability).all(), "non-finite V82 output")
    audit = {
        "train_n": len(train),
        "target_n": len(target),
        "numeric_features": len(NUMERIC_COLUMNS),
        "categorical_features": list(CATEGORICAL_COLUMNS),
        "category_sizes": processor.category_sizes,
        "mask_rate": MASK_RATE,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "head_epochs": HEAD_EPOCHS,
        "latent_dim": LATENT_DIM,
        "pretraining_target": "masked numeric/categorical reconstruction + dual-mask latent consistency",
        "pretraining_direction_labels_used": False,
        "pretraining_return_labels_used": False,
        "target_rows_used_in_pretraining": False,
        "encoder_frozen_before_direction_head": True,
        "direction_head": "single GPU linear layer",
        "final_pretraining_losses": history[-1],
        "first_pretraining_losses": history[0],
        "train_reconstruction_error_median": reference_error,
        "target_reconstruction_error_mean": float(np.mean(target_error)),
        "target_reliability_mean": float(np.mean(reliability)),
        "device": torch.cuda.get_device_name(device),
    }
    del head, model
    torch.cuda.empty_cache()
    return {"self_supervised_probability": probability, "reliability": reliability}, audit


def architecture_probability(
    baseline: np.ndarray,
    expert: dict[str, np.ndarray],
    architecture: dict[str, Any],
) -> np.ndarray:
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    expert_logit = logit(np.clip(expert["self_supervised_probability"], 1e-5, 1.0 - 1e-5))
    if architecture["mode"] == "fixed":
        weight: float | np.ndarray = float(architecture["weight"])
    elif architecture["mode"] == "reliability":
        weight = float(architecture["weight"]) * expert["reliability"]
    else:
        raise ValueError(f"unknown architecture mode: {architecture['mode']}")
    probability = expit((1.0 - weight) * base_logit + weight * expert_logit)
    require(np.isfinite(probability).all(), "V82 architecture returned non-finite probability")
    return probability


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) > 0 and len(valid) > 0, f"{label}: empty split")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: 35-minute embargo failed")
    overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    require(not overlap, f"{label}: event-group overlap")
    return {
        "train_n": len(train),
        "valid_n": len(valid),
        "train_end": train_end.isoformat(),
        "valid_start": valid_start.isoformat(),
        "strict_35m_embargo": True,
        "event_group_overlap": 0,
    }


def inner_partition(
    data_market: pd.DataFrame,
    champion_market: pd.DataFrame,
    outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible = champion_market.loc[champion_market.event_time_utc < outer_start - EMBARGO].copy()
    eligible = eligible.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(eligible) >= 250, "insufficient prior market OOF evidence")
    boundary = max(100, int(math.floor(len(eligible) * (1.0 - INNER_VALID_FRACTION))))
    valid_ids = set(eligible.iloc[boundary:].event_id)
    lookup = data_market.set_index("event_id")
    inner_valid = lookup.loc[list(valid_ids)].reset_index()
    inner_valid = inner_valid.sort_values(["event_time_utc", "event_id"], kind="stable")
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = data_market.loc[data_market.event_time_utc < inner_start - EMBARGO].copy()
    valid_groups = set(inner_valid.event_group_id.astype(str))
    inner_train = inner_train.loc[~inner_train.event_group_id.astype(str).isin(valid_groups)].copy()
    audit = chronology(inner_train, inner_valid, "inner")
    require(inner_valid.y.nunique() == 2, "inner valid is single-class")
    audit["inner_valid_fraction_target"] = INNER_VALID_FRACTION
    audit["inner_valid_is_prior_chronological_oof"] = True
    return inner_train, inner_valid, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    champion_lookup: pd.DataFrame,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    baseline = champion_lookup.set_index("event_id").loc[inner_valid.event_id, "prob"].to_numpy(float)
    baseline_metrics = metric(inner_valid, baseline)
    expert, expert_audit = fit_self_supervised_direction(
        inner_train, inner_valid, device, seed
    )
    trials: list[dict[str, Any]] = [
        {
            "name": "V69_NOOP",
            "architecture": None,
            "metrics": baseline_metrics,
            "auc_delta": 0.0,
            "balanced_accuracy_delta": 0.0,
            "eligible": True,
            "score": float(2.0 * baseline_metrics["auc"] + baseline_metrics["balanced_accuracy"]),
        }
    ]
    predictions: dict[str, np.ndarray] = {}
    for architecture in ARCHITECTURES:
        probability = architecture_probability(baseline, expert, architecture)
        predictions[architecture["name"]] = probability
        values = metric(inner_valid, probability)
        auc_delta = float(values["auc"] - baseline_metrics["auc"])
        ba_delta = float(values["balanced_accuracy"] - baseline_metrics["balanced_accuracy"])
        trials.append(
            {
                "name": architecture["name"],
                "architecture": architecture,
                "metrics": values,
                "auc_delta": auc_delta,
                "balanced_accuracy_delta": ba_delta,
                "eligible": ba_delta >= -0.010,
                "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
            }
        )
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past max 2*AUC+BA, BA delta >= -0.010; fixed blends only",
        "baseline": baseline_metrics,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "no_epoch_mask_or_blend_tuning": True,
    }, predictions, expert_audit


def run_nested(
    data: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    device = torch.device("cuda:0")
    diagnostic = champion.copy()
    diagnostic["v82_applied"] = False
    diagnostic["v82_architecture"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    stop = False
    for market in ("US", "KR"):
        data_market = data.loc[data.market.eq(market)].copy()
        champion_market = champion.loc[champion.market.eq(market)].copy()
        data_lookup = data_market.set_index("event_id")
        for fold in (2, 3, 4):
            outer_champion = champion_market.loc[champion_market.fold.eq(fold)].copy()
            outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
            require(len(outer_champion) >= 100, f"{market} fold {fold}: OOF support insufficient")
            outer_valid = data_lookup.loc[outer_champion.event_id].reset_index()
            outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
            outer_train = data_market.loc[data_market.event_time_utc < outer_start - EMBARGO].copy()
            valid_groups = set(outer_valid.event_group_id.astype(str))
            outer_train = outer_train.loc[~outer_train.event_group_id.astype(str).isin(valid_groups)].copy()
            outer_audit = chronology(outer_train, outer_valid, f"outer {market} fold {fold}")
            inner_train, inner_valid, inner_audit = inner_partition(
                data_market, champion_market, outer_start
            )
            policy, _, inner_expert_audit = choose_inner(
                inner_train, inner_valid, champion_market, device, SEED + 100 * fold + (0 if market == "US" else 50)
            )
            selected = policy["selected"]
            baseline_outer = outer_champion.prob.to_numpy(float)
            outer_expert_audit: dict[str, Any] = {"skipped_for_exact_noop": True}
            if selected["architecture"] is None:
                outer_probability = baseline_outer.copy()
            else:
                expert, outer_expert_audit = fit_self_supervised_direction(
                    outer_train, outer_valid, device,
                    SEED + 1000 + 100 * fold + (0 if market == "US" else 50),
                )
                outer_probability = architecture_probability(
                    baseline_outer, expert, selected["architecture"]
                )
            baseline_metrics = metric(outer_valid, baseline_outer)
            candidate_metrics = metric(outer_valid, outer_probability)
            probability_map = dict(zip(outer_valid.event_id, outer_probability))
            positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
            diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
            diagnostic.loc[positions, "v82_applied"] = selected["architecture"] is not None
            diagnostic.loc[positions, "v82_architecture"] = selected["name"]
            evidence = outer_champion[
                [
                    "event_id", "event_group_id", "event_time_utc", "market", "ticker",
                    "source_family", "fold", "y", "fwd_ret_30m", "high_conf",
                ]
            ].copy()
            evidence["baseline_prob"] = baseline_outer
            evidence["candidate_prob"] = outer_probability
            evidence["baseline_direction"] = baseline_outer >= 0.5
            evidence["candidate_direction"] = outer_probability >= 0.5
            evidence["v82_architecture"] = selected["name"]
            evidence_parts.append(evidence)
            audits.append(
                {
                    "market": market,
                    "fold": fold,
                    "outer_chronology": outer_audit,
                    "inner_chronology": inner_audit,
                    "policy": policy,
                    "inner_expert_audit": inner_expert_audit,
                    "outer_expert_audit": outer_expert_audit,
                    "outer_baseline": baseline_metrics,
                    "outer_candidate": candidate_metrics,
                    "outer_auc_delta": candidate_metrics["auc"] - baseline_metrics["auc"],
                    "outer_balanced_accuracy_delta": (
                        candidate_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
                    ),
                    "policy_locked_before_outer_evaluation": True,
                    "outer_labels_used_for_selection": False,
                }
            )
            print(
                f"[V82] market={market} fold={fold} architecture={selected['name']} "
                f"inner_auc_delta={selected['auc_delta']:+.6f} "
                f"outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
                flush=True,
            )
            if smoke:
                stop = True
                break
        if stop:
            break
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "outer evidence repeats an event")
    return diagnostic, evidence, audits


def paired_date_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    work["date"] = pd.to_datetime(work.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby(["market", "fold", "date"], sort=True)]
    require(len(blocks) >= 10, "too few outer date blocks")
    generator = np.random.default_rng(SEED)
    auc_deltas: list[float] = []
    ba_deltas: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_deltas.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)))
        ba_deltas.append(
            float(
                balanced_accuracy_score(y, candidate >= 0.5)
                - balanced_accuracy_score(y, baseline >= 0.5)
            )
        )
    require(len(auc_deltas) >= int(0.95 * draws), "bootstrap lost too many samples")
    return {
        "method": "paired market/fold/date-block bootstrap over locked self-supervised architectures",
        "seed": SEED,
        "requested_draws": draws,
        "effective_draws": len(auc_deltas),
        "date_blocks": len(blocks),
        "auc_delta_probability_gt_zero": float(np.mean(np.asarray(auc_deltas) > 0.0)),
        "auc_delta_quantiles": {
            str(q): float(np.quantile(auc_deltas, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
        "balanced_accuracy_delta_probability_gt_zero": float(np.mean(np.asarray(ba_deltas) > 0.0)),
        "balanced_accuracy_delta_quantiles": {
            str(q): float(np.quantile(ba_deltas, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
) -> dict[str, Any]:
    baseline_report = json.loads(V69_ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    candidate_summary = v44.summarize(diagnostic)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V82 canonical gate mismatch")
    outer_baseline = metric(evidence, evidence.baseline_prob.to_numpy(float))
    outer_candidate = metric(evidence, evidence.candidate_prob.to_numpy(float))
    bootstrap = paired_date_bootstrap(evidence, draws)
    fold_auc_deltas = np.asarray([audit["outer_auc_delta"] for audit in audits], dtype=float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(
        champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)
    ) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool))
    material_checks = {
        "outer_evidence_auc_delta_gt_0_003": outer_candidate["auc"] - outer_baseline["auc"] > 0.003,
        "outer_evidence_balanced_accuracy_delta_gt_0": (
            outer_candidate["balanced_accuracy"] - outer_baseline["balanced_accuracy"] > 0.0
        ),
        "positive_outer_block_auc_fraction_ge_2_of_3": float(np.mean(fold_auc_deltas > 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta_probability_gt_zero"] >= 0.75,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": (
            candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001
        ),
        "hc_accuracy_delta_ge_minus_0_005": (
            candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005
        ),
        "hc_net_delta_ge_minus_0_0005": (
            candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005
        ),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_blocks": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"]
            for audit in audits
        ),
        "pretraining_label_free_all_blocks": all(
            not audit["inner_expert_audit"]["pretraining_direction_labels_used"]
            and not audit["inner_expert_audit"]["pretraining_return_labels_used"]
            and not audit["inner_expert_audit"]["target_rows_used_in_pretraining"]
            for audit in audits
        ),
        "encoder_frozen_before_head_all_blocks": all(
            audit["inner_expert_audit"]["encoder_frozen_before_direction_head"] for audit in audits
        ),
    }
    material_pass = bool(all(material_checks.values()))
    selected = diagnostic if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(selected_summary["research_gate"] == canonical_selected, "selected canonical gate mismatch")
    exact_entire_frame = selected.equals(champion)
    require(material_pass or exact_entire_frame, "V82 failure did not restore complete V69 frame")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_V69_NOOP",
        "material_pass": material_pass,
        "material_checks": material_checks,
        "outer_baseline": outer_baseline,
        "outer_candidate": outer_candidate,
        "outer_auc_delta": outer_candidate["auc"] - outer_baseline["auc"],
        "outer_balanced_accuracy_delta": (
            outer_candidate["balanced_accuracy"] - outer_baseline["balanced_accuracy"]
        ),
        "positive_outer_block_auc_fraction": float(np.mean(fold_auc_deltas > 0.0)),
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline,
            "candidate": canonical_candidate,
            "selected": canonical_selected,
            "reported_equals_controller_recomputed": True,
        },
        "direction_change": {
            "probability_changed_n": int(
                np.sum(champion.prob.to_numpy(float) != diagnostic.prob.to_numpy(float))
            ),
            "direction_changed_n": int(
                np.sum(
                    (champion.prob.to_numpy(float) >= 0.5)
                    != (diagnostic.prob.to_numpy(float) >= 0.5)
                )
            ),
            "baseline_probability_sha256": array_sha256(champion.prob.to_numpy(float)),
            "candidate_probability_sha256": array_sha256(diagnostic.prob.to_numpy(float)),
            "selected_probability_sha256": array_sha256(selected.prob.to_numpy(float)),
            "explicit_direction_challenger": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact complete V69 frame" if not material_pass else None,
            "exact_entire_frame_verified": bool(material_pass or exact_entire_frame),
        },
        "diagnostic_frame": diagnostic,
        "selected_frame": selected,
    }


def build_reports(
    champion: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected_contract = dict(evaluation["selected_summary"])
    selected_contract["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"],
    }
    validation = (
        "Immutable V36 DEV and atomic V69 only; cut-local label-free masked numeric/categorical reconstruction "
        "plus dual-mask latent consistency; frozen encoder; GPU linear direction head; market-specific outer "
        "folds 2-4; strict 35-minute nested chronology; inner-only fixed architecture selection."
    )
    report = {
        "version": "V82",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "material_change": "self-supervised masked causal-state encoder frozen before supervised direction head",
        "disjoint_from": {
            "V48_V75_V78": "not supervised boosting, multiclass, or pairwise ranking",
            "V50_V52": "not a frozen externally pretrained text embedding or supervised embedding MLP",
            "V55": "not a supervised temporal CNN",
            "V77_V81": "not issuer balancing or worst-environment Group-DRO",
        },
        "pretraining": {
            "mask_rate": MASK_RATE,
            "epochs": PRETRAIN_EPOCHS,
            "latent_dim": LATENT_DIM,
            "consistency_weight": CONSISTENCY_WEIGHT,
            "categorical_loss_weight": CATEGORICAL_LOSS_WEIGHT,
            "direction_or_return_labels_used": False,
        },
        "architectures": list(ARCHITECTURES),
        "fold_audits": audits,
        "input_audit": input_audit,
        "evaluation": compact,
        "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V82",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "opened_dev_only": True,
        "reserved_outcomes_loaded": False,
        "selected": selected_contract,
        "candidate": evaluation["candidate_summary"],
        "baseline": evaluation["baseline_summary"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    gate_audit = {
        "version": "V82",
        "status": "MATCH",
        "reported": evaluation["selected_summary"]["research_gate"],
        "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        "material_gate": selected_contract["material_gate"],
    }
    model_comparison = {
        "version": "V82",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "models": {
            "V69_CHAMPION": evaluation["baseline_summary"],
            "V82_MASKED_STATE_DIAGNOSTIC": evaluation["candidate_summary"],
            "V82_FAIL_CLOSED_SELECTED": selected_contract,
        },
        "evaluation": compact,
        "input_audit": input_audit,
        "fold_audits": audits,
        "seal_authorized": False,
    }
    source_transfer = {
        "version": "V82",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
        "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
        "selected_by_market": selected_contract["metrics"]["by_market"],
        "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
        "outer": {
            "auc_delta": evaluation["outer_auc_delta"],
            "balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
            "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta_probability_gt_zero"],
        },
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    fold_rows = []
    for audit in audits:
        chosen = audit["policy"]["selected"]
        fold_rows.append(
            {
                "market": audit["market"],
                "fold": audit["fold"],
                "outer_train_n": audit["outer_chronology"]["train_n"],
                "outer_valid_n": audit["outer_chronology"]["valid_n"],
                "inner_train_n": audit["inner_chronology"]["train_n"],
                "inner_valid_n": audit["inner_chronology"]["valid_n"],
                "selected_architecture": chosen["name"],
                "inner_auc_delta": chosen["auc_delta"],
                "inner_ba_delta": chosen["balanced_accuracy_delta"],
                "outer_auc_delta": audit["outer_auc_delta"],
                "outer_ba_delta": audit["outer_balanced_accuracy_delta"],
                "strict_35m_embargo": True,
                "outer_labels_used_for_selection": False,
            }
        )
    json_reports = {
        "MODEL_COMPARISON.json": model_comparison,
        "V82_MASKED_STATE_SELF_SUPERVISED_REPORT.json": report,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": source_transfer,
        "CANONICAL_RESEARCH_GATE_AUDIT.json": gate_audit,
        "BOOTSTRAP_DIRECTION_DELTA_REPORT.json": evaluation["bootstrap"],
    }
    selected_columns = list(champion.columns)
    table_reports = {
        "V82_SELF_SUPERVISED_FOLD_AUDIT.csv": pd.DataFrame(fold_rows),
        "V82_SELF_SUPERVISED_DIRECTION_EVIDENCE_OOF.csv.gz": evidence,
        "V82_DIAGNOSTIC_CHALLENGER_OOF.csv.gz": evaluation["diagnostic_frame"],
        "V82_FAIL_CLOSED_SELECTED_OOF.csv.gz": evaluation["selected_frame"][selected_columns],
    }
    return json_reports, table_reports


def write_outputs(
    json_reports: dict[str, Any],
    table_reports: dict[str, pd.DataFrame],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in json_reports.items():
        atomic_json(payload, OUT / name)
    for name, frame in table_reports.items():
        atomic_csv(frame, OUT / name)
    material_spec = {
        "hypothesis": HYPOTHESIS,
        "mask_rate": MASK_RATE,
        "latent_dim": LATENT_DIM,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "head_epochs": HEAD_EPOCHS,
        "consistency_weight": CONSISTENCY_WEIGHT,
        "architectures": ARCHITECTURES,
        "embargo_seconds": int(EMBARGO.total_seconds()),
        "validation": "nested chronological cut-local self-supervised OOF",
    }
    experiment_id = hashlib.sha256(
        (
            "|".join(input_audit["input_sha256"].values())
            + "|"
            + json.dumps(material_spec, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    status = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "phase": (
            "ROBUST_SURVIVOR"
            if evaluation["selected_summary"]["research_gate"]["robust_survivor"]
            else "RESEARCH_FAIL"
        ),
        "canonical_gate_passed": evaluation["selected_summary"]["research_gate"]["passed"],
        "canonical_gate_total": evaluation["selected_summary"]["research_gate"]["total"],
        "material_pass": evaluation["material_pass"],
        "champion_changed": evaluation["material_pass"],
        "completed_at": now(),
        "experiment_id": experiment_id,
        "reserved_state": "UNOPENED",
        "seal_authorized": False,
        "output_scope": str(OUT.relative_to(ROOT)) if OUT.is_relative_to(ROOT) else str(OUT),
    }
    atomic_json(status, OUT / "RUN_STATUS.json")
    files: dict[str, Any] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}:
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"version": VERSION, "hypothesis": HYPOTHESIS, "experiment_id": experiment_id, "files": files}
    atomic_json(manifest, OUT / "ARTIFACT_MANIFEST.json")
    commit = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "committed_at": now(),
        "experiment_id": experiment_id,
        "input_sha256": input_audit["input_sha256"],
        "material_fingerprint": hashlib.sha256(
            json.dumps(material_spec, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
    }
    atomic_json(commit, OUT / "COMMIT.json")
    return {"experiment_id": experiment_id, "artifact_count": len(files) + 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    data, champion, input_audit = verify_and_load()
    if args.audit_only:
        print(
            json.dumps(
                clean(
                    {
                        "status": "AUDIT_OK",
                        "hypothesis": HYPOTHESIS,
                        "input_audit": input_audit,
                        "self_supervised_spec": {
                            "numeric_features": len(NUMERIC_COLUMNS),
                            "categorical_features": list(CATEGORICAL_COLUMNS),
                            "mask_rate": MASK_RATE,
                            "latent_dim": LATENT_DIM,
                            "pretrain_epochs": PRETRAIN_EPOCHS,
                            "head_epochs": HEAD_EPOCHS,
                            "pretraining_uses_labels": False,
                        },
                        "architectures": list(ARCHITECTURES),
                        "output_written": False,
                    }
                ),
                indent=2,
            )
        )
        return
    diagnostic, evidence, audits = run_nested(data, champion, smoke=args.smoke_test)
    evaluation = evaluate(
        champion,
        diagnostic,
        evidence,
        audits,
        draws=250 if args.smoke_test else BOOTSTRAP_DRAWS,
    )
    summary = {
        "status": evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "material_pass": evaluation["material_pass"],
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "inner_trials": [audit["policy"]["trials"] for audit in audits],
        "self_supervised_audits": [
            {
                "market": audit["market"],
                "fold": audit["fold"],
                "inner": audit["inner_expert_audit"],
                "outer": audit["outer_expert_audit"],
            }
            for audit in audits
        ],
        "outer_baseline": evaluation["outer_baseline"],
        "outer_candidate": evaluation["outer_candidate"],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
        "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta_probability_gt_zero"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    if args.smoke_test:
        print(json.dumps(clean({**summary, "status": "SMOKE_OK", "output_written": False}), indent=2))
        return
    json_reports, table_reports = build_reports(champion, evidence, audits, evaluation, input_audit)
    if args.dry_run:
        print(json.dumps(clean({**summary, "status": "DRY_RUN_OK", "output_written": False}), indent=2))
        return
    write_result = write_outputs(json_reports, table_reports, evaluation, input_audit)
    print(json.dumps(clean({**summary, "output_dir": str(OUT), "write": write_result}), indent=2))


if __name__ == "__main__":
    main()
