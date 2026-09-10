"""V221+ leakage-safe contribution-polarity diagnostic.

This is a research harness, not a production intervention.  It conditions on
the frozen V69 OOF anchors and estimates a feature block's effective
contribution with paired, independently fitted full and leave-block-out
models.  For every validation row::

    delta_block = z_full - z_without_block
    z_alpha = z_without_block + alpha * delta_block

No input feature is sign-flipped.  Alpha selection is made only from purged,
embargoed inner folds.  The selected policy is frozen before an outer fold is
evaluated, and outer outcomes are never used to reselect it.

All artifacts are confined to ``research/audits`` (or ``staging`` when an
explicit output directory is supplied).  The script never opens a research
seal/final reserve and never modifies V69, V220, registries, queues, or state.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Bound native libraries before importing numpy/scikit-learn.  main() also
# applies an OS affinity mask.  GPU use is intentionally hidden/disabled.
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "research" / "FEATURE_BLOCK_REGISTRY_V221PLUS.json"
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_PATH = ROOT / "output_V69" / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
DEFAULT_AUDIT_ROOT = ROOT / "research" / "audits" / "V221PLUS_CONTRIBUTION_POLARITY"

VERSION = "V221PLUS_COUNTERFACTUAL_BLOCK_POLARITY_V1"
COST = 0.002
EMBARGO = pd.Timedelta(minutes=35)
BASE_ALPHAS = (-1.0, 0.0, 1.0)
DOSAGE_ALPHAS = (-1.0, 0.0, 0.5, 1.0, 1.5)
ROLE_ORDER = ("direction", "confidence", "opportunity")
SOURCE_ORDER = ("US_SEC", "US_NEWS", "KR_NEWS", "KR_KIND")
ALPHA_LABELS = {
    -1.0: "alpha_minus1", 0.0: "alpha_zero", 0.5: "alpha_plus05",
    1.0: "alpha_plus1", 1.5: "alpha_plus15",
}
EXTREME_BANDS = (
    (0.00, 0.50, "0-50%"), (0.50, 0.80, "50-80%"),
    (0.80, 0.90, "80-90%"), (0.90, 0.95, "90-95%"),
    (0.95, 1.00, "95-100%"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(score, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def counterfactual_score(z_full: np.ndarray, z_without: np.ndarray, alpha: float) -> np.ndarray:
    """Contribution intervention; it never transforms an input feature."""
    full = np.asarray(z_full, dtype=float)
    without = np.asarray(z_without, dtype=float)
    require(full.shape == without.shape, "full/ablated score shapes differ")
    return without + float(alpha) * (full - without)


def configure_runtime(cpu_ids: Sequence[int]) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(int(value) for value in cpu_ids))
    require(requested, "at least one CPU id is required")
    available_count = os.cpu_count() or 1
    require(all(0 <= value < available_count for value in requested),
            f"requested CPUs {requested} unavailable; logical CPU count={available_count}")
    numeric_threads = len(requested)
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(numeric_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    if sys.platform == "win32":
        mask = sum(1 << value for value in requested)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(requested))
    else:
        raise RuntimeError("strict CPU affinity is unsupported on this platform")
    return {
        "cpu_ids": list(requested), "thread_count": numeric_threads,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "nvidia_visible_devices": os.environ["NVIDIA_VISIBLE_DEVICES"],
        "gpu_hidden": True,
    }


def parse_cpu_ids(value: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())


def load_registry() -> tuple[dict[str, Any], str]:
    require(REGISTRY_PATH.is_file(), f"missing registry: {REGISTRY_PATH}")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    require(registry.get("authority", {}).get("selection_scope") == "INNER_PAST_ONLY",
            "registry selection scope is not INNER_PAST_ONLY")
    require(registry.get("authority", {}).get("outer_labels_for_selection") is False,
            "registry permits outer-label selection")
    require(registry.get("counterfactual", {}).get("raw_feature_sign_flip_allowed") is False,
            "registry unexpectedly permits raw feature sign flips")
    require(tuple(float(v) for v in registry["counterfactual"]["limited_dosage_grid"]) == DOSAGE_ALPHAS,
            "registry dosage grid changed")
    return registry, sha256(REGISTRY_PATH)


def verify_and_load_inputs(registry: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    require(DEV_PATH.is_file(), f"missing immutable DEV: {DEV_PATH}")
    require(V69_PATH.is_file(), f"missing frozen V69 OOF: {V69_PATH}")
    hashes = {"dev": sha256(DEV_PATH), "v69_oof": sha256(V69_PATH)}
    authority = registry["authority"]
    require(hashes["dev"] == authority["immutable_v36_dev_sha256"],
            "immutable DEV hash differs from registry authority")
    require(hashes["v69_oof"] == authority["atomic_v69_oof_sha256"],
            "V69 OOF hash differs from registry authority")

    dev = pd.read_csv(DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    oof = pd.read_csv(V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    require(dev.event_id.is_unique and oof.event_id.is_unique, "input event_id must be unique")
    require(set(oof.event_id).issubset(set(dev.event_id)), "V69 OOF is not a DEV subset")
    required_oof = {
        "event_id", "event_group_id", "event_time_utc", "source_family", "y",
        "fwd_ret_30m", "prob", "confidence_signal", "high_conf",
        "opportunity_confidence", "opportunity_high", "cluster_weight",
    }
    require(required_oof.issubset(oof.columns),
            f"V69 OOF columns missing: {sorted(required_oof - set(oof.columns))}")
    aligned = dev.set_index("event_id").loc[oof.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), oof.y.to_numpy(int)),
            "V69 labels differ from immutable DEV")
    require(np.allclose(aligned.fwd_ret_30m.to_numpy(float),
                        oof.fwd_ret_30m.to_numpy(float), rtol=0.0, atol=1e-15),
            "V69 returns differ from immutable DEV")
    require(np.array_equal(aligned.source_family.astype(str).to_numpy(),
                           oof.source_family.astype(str).to_numpy()),
            "V69 source identities differ from immutable DEV")
    for column in oof.columns:
        if column not in aligned.columns or column in {
            "event_group_id", "event_time_utc", "y", "fwd_ret_30m", "source_family",
        }:
            aligned[f"v69_{column}" if column in aligned.columns else column] = oof[column].to_numpy()
    # Stable canonical names for OOF anchors, regardless of overlaps with DEV.
    for column in required_oof - {"event_id", "event_group_id", "event_time_utc",
                                  "source_family", "y", "fwd_ret_30m"}:
        aligned[f"v69_{column}"] = oof[column].to_numpy()
    aligned["event_time_utc"] = pd.to_datetime(aligned.event_time_utc, utc=True)
    for column in ("v69_prob", "v69_confidence_signal", "v69_opportunity_confidence"):
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
        require(aligned[column].between(0.0, 1.0).all(), f"invalid OOF anchor: {column}")
    for column in ("v69_high_conf", "v69_opportunity_high"):
        aligned[column] = aligned[column].map(
            lambda value: value if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"1", "true", "yes"}
        ).astype(bool)
    aligned["cluster_weight"] = pd.to_numeric(
        aligned.get("v69_cluster_weight", oof.cluster_weight), errors="coerce"
    ).fillna(1.0).clip(lower=1e-6)
    aligned["direction_correct"] = (
        (aligned.v69_prob.to_numpy(float) >= 0.5) == aligned.y.to_numpy(int)
    ).astype(int)
    aligned["opportunity_target"] = np.log1p(
        np.abs(aligned.fwd_ret_30m.to_numpy(float)) * 10000.0
    )
    audit = {
        "authorized_inputs": [str(DEV_PATH.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT))],
        "sha256": hashes, "dev_rows": int(len(dev)), "v69_oof_rows": int(len(oof)),
        "source_rows": {str(k): int(v) for k, v in aligned.source_family.value_counts().items()},
        "dev_extension_loaded": False, "research_seal_loaded": False,
        "final_meta_loaded": False, "v69_changed": False, "v220_changed": False,
    }
    return aligned, audit


def semantic_text(frame: pd.DataFrame) -> pd.Series:
    # Label-independent text block.  Prefixes are known at event time and the
    # body cap bounds memory without selecting tokens from any outcome.
    source = frame.source_family.fillna("").astype(str)
    form = frame.get("form", pd.Series("", index=frame.index)).fillna("").astype(str)
    event = frame.get("event_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    headline = frame.get("headline", pd.Series("", index=frame.index)).fillna("").astype(str)
    body = frame.get("body", pd.Series("", index=frame.index)).fillna("").astype(str).str.slice(0, 5000)
    return "source_" + source + " form_" + form + " event_" + event + " " + headline + " " + body


def prepare_features(frame: pd.DataFrame, registry: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result.event_time_utc, utc=True)
    result["event_hour"] = timestamp.dt.hour + timestamp.dt.minute / 60.0
    result["event_weekday"] = timestamp.dt.weekday
    result["event_month"] = timestamp.dt.month
    result["event_hour_sin"] = np.sin(2.0 * np.pi * result.event_hour / 24.0)
    result["event_hour_cos"] = np.cos(2.0 * np.pi * result.event_hour / 24.0)
    for suffix in ("2", "5", "15", "30", "60"):
        result[f"abs_pre_ret_{suffix}"] = pd.to_numeric(
            result[f"pre_ret_{suffix}"], errors="coerce"
        ).abs()
    result["ret_slope_2_15"] = (
        pd.to_numeric(result.pre_ret_2, errors="coerce")
        - pd.to_numeric(result.pre_ret_15, errors="coerce")
    )
    result["ret_slope_5_30"] = (
        pd.to_numeric(result.pre_ret_5, errors="coerce")
        - pd.to_numeric(result.pre_ret_30, errors="coerce")
    )
    result["kw_sentiment"] = (
        pd.to_numeric(result.event_positive_kw, errors="coerce")
        - pd.to_numeric(result.event_negative_kw, errors="coerce")
        - pd.to_numeric(result.event_financing_kw, errors="coerce")
    )
    for suffix in ("5", "15", "30", "60"):
        result[f"excess_ret_{suffix}"] = (
            pd.to_numeric(result[f"pre_ret_{suffix}"], errors="coerce")
            - pd.to_numeric(result[f"benchmark_ret_{suffix}"], errors="coerce")
        )
    result["semantic_text"] = semantic_text(result)
    registered: list[str] = []
    forbidden = set(registry["forbidden_label_or_future"])
    for block, spec in registry["blocks"].items():
        columns = list(spec["columns"])
        require(not (set(columns) & forbidden), f"block {block} contains future/label columns")
        require(set(columns).issubset(result.columns),
                f"block {block} unavailable columns: {sorted(set(columns) - set(result.columns))}")
        registered.extend(columns)
    require(len(registered) == len(set(registered)), "feature block registry overlaps")
    return result


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    fold: int


def expanding_purged_splits(
    frame: pd.DataFrame,
    folds: int,
    minimum_train: int,
    minimum_valid: int,
) -> list[ChronologicalSplit]:
    require(folds >= 1, "fold count must be positive")
    ordered = frame.sort_values(["event_time_utc", "event_id"], kind="mergesort").reset_index(drop=True)
    chunks = np.array_split(np.arange(len(ordered)), folds + 1)
    output: list[ChronologicalSplit] = []
    for fold in range(1, folds + 1):
        if not len(chunks[fold]):
            continue
        valid = ordered.iloc[chunks[fold]].copy()
        boundary = pd.Timestamp(valid.event_time_utc.min())
        train = ordered[ordered.event_time_utc < boundary - EMBARGO].copy()
        group_column = "event_group_id" if "event_group_id" in ordered.columns else "event_id"
        valid_groups = set(valid[group_column].fillna(valid.event_id).astype(str))
        train = train[~train[group_column].fillna(train.event_id).astype(str).isin(valid_groups)].copy()
        if len(train) < minimum_train or len(valid) < minimum_valid:
            continue
        output.append(ChronologicalSplit(train.reset_index(drop=True), valid.reset_index(drop=True), fold))
    return output


def chronology_audit(split: ChronologicalSplit) -> dict[str, Any]:
    train_end = pd.Timestamp(split.train.event_time_utc.max())
    valid_start = pd.Timestamp(split.valid.event_time_utc.min())
    group_column = "event_group_id" if "event_group_id" in split.train.columns else "event_id"
    overlap = set(split.train[group_column].astype(str)) & set(split.valid[group_column].astype(str))
    require(train_end < valid_start - EMBARGO, "chronological split violates 35-minute embargo")
    require(not overlap, "chronological split leaks an event group")
    return {
        "fold": split.fold, "train_n": len(split.train), "valid_n": len(split.valid),
        "train_end": train_end.isoformat(), "valid_start": valid_start.isoformat(),
        "embargo_minutes": 35, "event_group_overlap": 0,
    }


def fit_block_matrices(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    registry: dict[str, Any],
    blocks: Sequence[str],
    max_text_features: int,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, sparse.csr_matrix], dict[str, int]]:
    x_train = prepare_features(train, registry)
    x_valid = prepare_features(valid, registry)
    train_matrices: dict[str, sparse.csr_matrix] = {}
    valid_matrices: dict[str, sparse.csr_matrix] = {}
    dimensions: dict[str, int] = {}
    for block in blocks:
        spec = registry["blocks"][block]
        columns = list(spec["columns"])
        if spec["kind"] == "numeric":
            imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
            train_values = imputer.fit_transform(x_train[columns])
            valid_values = imputer.transform(x_valid[columns])
            scaler = StandardScaler(with_mean=False)
            train_matrix = sparse.csr_matrix(scaler.fit_transform(train_values), dtype=np.float32)
            valid_matrix = sparse.csr_matrix(scaler.transform(valid_values), dtype=np.float32)
        elif spec["kind"] == "categorical":
            encoder = OneHotEncoder(
                handle_unknown="ignore", min_frequency=2, sparse_output=True, dtype=np.float32,
            )
            train_matrix = encoder.fit_transform(x_train[columns].fillna("").astype(str)).tocsr()
            valid_matrix = encoder.transform(x_valid[columns].fillna("").astype(str)).tocsr()
        elif spec["kind"] == "text":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.997,
                max_features=max_text_features, sublinear_tf=True, dtype=np.float32,
                token_pattern=r"(?u)\b\w+\b",
            )
            try:
                train_matrix = vectorizer.fit_transform(x_train[columns[0]].fillna("")).tocsr()
                valid_matrix = vectorizer.transform(x_valid[columns[0]].fillna("")).tocsr()
            except ValueError as error:
                if "empty vocabulary" not in str(error).lower():
                    raise
                train_matrix = sparse.csr_matrix((len(train), 1), dtype=np.float32)
                valid_matrix = sparse.csr_matrix((len(valid), 1), dtype=np.float32)
        else:
            raise RuntimeError(f"unsupported block kind: {spec['kind']}")
        require(train_matrix.shape[1] == valid_matrix.shape[1], f"matrix width mismatch: {block}")
        train_matrices[block] = train_matrix
        valid_matrices[block] = valid_matrix
        dimensions[block] = int(train_matrix.shape[1])
    return train_matrices, valid_matrices, dimensions


def anchor_matrix(frame: pd.DataFrame, role: str) -> sparse.csr_matrix:
    direction_logit = logit(frame.v69_prob.to_numpy(float))
    if role == "direction":
        values = np.column_stack([direction_logit])
    elif role == "confidence":
        component_probabilities = [
            column for column in frame.columns
            if column.startswith("prob_") and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if component_probabilities:
            components = frame[component_probabilities].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            disagreement = np.nanstd(components, axis=1)
        else:
            disagreement = np.zeros(len(frame), dtype=float)
        values = np.column_stack([
            logit(frame.v69_confidence_signal.to_numpy(float)),
            np.abs(direction_logit), disagreement,
        ])
    elif role == "opportunity":
        values = np.column_stack([
            logit(frame.v69_opportunity_confidence.to_numpy(float)), np.abs(direction_logit),
        ])
    else:
        raise RuntimeError(f"unknown role: {role}")
    values = np.nan_to_num(values, nan=0.0, posinf=35.0, neginf=-35.0)
    return sparse.csr_matrix(values, dtype=np.float32)


def stack_design(
    matrices: dict[str, sparse.csr_matrix],
    frame: pd.DataFrame,
    role: str,
    blocks: Sequence[str],
) -> sparse.csr_matrix:
    return sparse.hstack(
        [anchor_matrix(frame, role), *(matrices[block] for block in blocks)],
        format="csr", dtype=np.float32,
    )


def role_target(frame: pd.DataFrame, role: str) -> np.ndarray:
    if role == "direction":
        return frame.y.to_numpy(int)
    if role == "confidence":
        return frame.direction_correct.to_numpy(int)
    if role == "opportunity":
        return frame.opportunity_target.to_numpy(float)
    raise RuntimeError(f"unknown role: {role}")


def fit_role_score(
    x_train: sparse.csr_matrix,
    train: pd.DataFrame,
    x_valid: sparse.csr_matrix,
    role: str,
    seed: int,
) -> np.ndarray:
    target = role_target(train, role)
    weight = train.cluster_weight.to_numpy(float)
    if role in {"direction", "confidence"}:
        if np.unique(target).size < 2:
            probability = np.full(len(train), np.clip(float(np.mean(target)), 1e-6, 1 - 1e-6))
            return np.full(x_valid.shape[0], float(logit(probability[:1])[0]))
        model = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=3e-4, max_iter=900,
            tol=1e-3, average=True, class_weight="balanced", random_state=seed,
        )
        model.fit(x_train, target, sample_weight=weight)
        return np.asarray(model.decision_function(x_valid), dtype=float)
    model = SGDRegressor(
        loss="huber", epsilon=1.35, penalty="l2", alpha=3e-4,
        max_iter=1000, tol=1e-3, average=True, random_state=seed,
    )
    model.fit(x_train, target, sample_weight=weight)
    return np.asarray(model.predict(x_valid), dtype=float)


def paired_role_scores(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    train_matrices: dict[str, sparse.csr_matrix],
    valid_matrices: dict[str, sparse.csr_matrix],
    blocks: Sequence[str],
    seed: int,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {role: {} for role in ROLE_ORDER}
    for role_index, role in enumerate(ROLE_ORDER):
        full_train = stack_design(train_matrices, train, role, blocks)
        full_valid = stack_design(valid_matrices, valid, role, blocks)
        z_full = fit_role_score(full_train, train, full_valid, role, seed + 100 * role_index)
        for block_index, block in enumerate(blocks):
            kept = [candidate for candidate in blocks if candidate != block]
            ablated_train = stack_design(train_matrices, train, role, kept)
            ablated_valid = stack_design(valid_matrices, valid, role, kept)
            z_without = fit_role_score(
                ablated_train, train, ablated_valid, role,
                # Pair the stochastic optimization seed with the full model so
                # delta is not contaminated by a gratuitous RNG change.
                seed + 100 * role_index,
            )
            result[role][block] = (z_full, z_without)
    return result


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    return (
        float(roc_auc_score(target[mask], score[mask]))
        if mask.sum() >= 2 and np.unique(target[mask]).size == 2 else None
    )


def top_mask(score: np.ndarray, count: int) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    count = max(0, min(int(count), len(score)))
    mask = np.zeros(len(score), dtype=bool)
    if count:
        order = np.lexsort((np.arange(len(score)), -np.nan_to_num(score, nan=-np.inf)))
        mask[order[:count]] = True
    return mask


def direction_net(frame: pd.DataFrame, selector: np.ndarray | None = None) -> np.ndarray:
    signed = np.where(frame.v69_prob.to_numpy(float) >= 0.5, 1.0, -1.0)
    net = signed * frame.fwd_ret_30m.to_numpy(float) - COST
    return net if selector is None else net[np.asarray(selector, dtype=bool)]


def confidence_bin_monotonicity(target: np.ndarray, score: np.ndarray, bins: int = 5) -> float | None:
    work = pd.DataFrame({"target": target, "score": score}).sort_values(
        ["score"], kind="mergesort"
    ).reset_index(drop=True)
    if len(work) < bins:
        return None
    work["bin"] = np.minimum(bins - 1, np.arange(len(work)) * bins // len(work))
    accuracy = work.groupby("bin", sort=True).target.mean()
    return finite(pd.Series(np.arange(len(accuracy)), dtype=float).corr(accuracy, method="spearman"))


def metric_row(
    frame: pd.DataFrame,
    role: str,
    raw_score: np.ndarray,
    material_threshold: float,
) -> dict[str, Any]:
    score = np.asarray(raw_score, dtype=float)
    probability = sigmoid(score)
    y = frame.y.to_numpy(int)
    direction_prediction = probability >= 0.5
    direction_correct = (direction_prediction == y).astype(int)
    naive = max(float(y.mean()), 1.0 - float(y.mean()))
    anchor_high_count = int(frame.v69_high_conf.sum())
    opportunity_high_count = int(frame.v69_opportunity_high.sum())
    magnitude = np.abs(frame.fwd_ret_30m.to_numpy(float))
    material = (magnitude >= material_threshold).astype(int)

    if role == "direction":
        confidence = np.abs(probability - 0.5) * 2.0
        selected = top_mask(confidence, anchor_high_count)
        counterfactual_net = (
            np.where(direction_prediction, 1.0, -1.0)
            * frame.fwd_ret_30m.to_numpy(float) - COST
        )
        return {
            "n": len(frame), "primary_metric": safe_auc(y, probability),
            "auc": safe_auc(y, probability),
            "balanced_accuracy": (
                float(balanced_accuracy_score(y, direction_prediction))
                if np.unique(y).size == 2 else None
            ),
            "edge_vs_naive": float(direction_correct.mean() - naive),
            "accuracy": float(direction_correct.mean()),
            "confidence_correctness_auc": safe_auc(direction_correct, confidence),
            "hc_accuracy": float(direction_correct[selected].mean()) if selected.any() else None,
            "hc_coverage": float(selected.mean()),
            "hc_net": float(counterfactual_net[selected].mean()) if selected.any() else None,
            "material_move_auc": None, "magnitude_spearman": None,
            "confidence_monotonicity": confidence_bin_monotonicity(direction_correct, confidence),
        }
    if role == "confidence":
        target = frame.direction_correct.to_numpy(int)
        selected = top_mask(probability, anchor_high_count)
        order = np.argsort(-probability, kind="stable")
        top10 = order[:max(1, int(math.ceil(0.10 * len(order))))]
        top20 = order[:max(1, int(math.ceil(0.20 * len(order))))]
        return {
            "n": len(frame), "primary_metric": safe_auc(target, probability),
            "auc": None, "balanced_accuracy": None, "edge_vs_naive": None,
            "accuracy": float(((probability >= 0.5) == target).mean()),
            "confidence_correctness_auc": safe_auc(target, probability),
            "hc_accuracy": float(target[selected].mean()) if selected.any() else None,
            "hc_coverage": float(selected.mean()),
            "hc_net": float(direction_net(frame, selected).mean()) if selected.any() else None,
            "top10_accuracy": float(target[top10].mean()),
            "top20_accuracy": float(target[top20].mean()),
            "material_move_auc": None, "magnitude_spearman": None,
            "confidence_monotonicity": confidence_bin_monotonicity(target, probability),
        }
    selected = top_mask(score, opportunity_high_count)
    magnitude_spearman = finite(pd.Series(score).corr(pd.Series(magnitude), method="spearman"))
    material_auc = safe_auc(material, score)
    # Material-move AUC is primary; magnitude rank is the deterministic tie-break.
    return {
        "n": len(frame), "primary_metric": material_auc,
        "auc": None, "balanced_accuracy": None, "edge_vs_naive": None,
        "accuracy": float(((score >= np.median(score)) == material).mean()),
        "confidence_correctness_auc": None,
        "hc_accuracy": float(frame.direction_correct.to_numpy(int)[selected].mean()) if selected.any() else None,
        "hc_coverage": float(selected.mean()),
        "hc_net": float(direction_net(frame, selected).mean()) if selected.any() else None,
        "material_move_auc": material_auc, "magnitude_spearman": magnitude_spearman,
        "confidence_monotonicity": None,
    }


def selection_key(metrics: dict[str, Any], role: str, alpha: float) -> tuple[float, ...]:
    def value(name: str, fallback: float = -1e9) -> float:
        candidate = finite(metrics.get(name))
        return fallback if candidate is None else candidate

    # The last values impose deterministic, conservative ties: +1, then 0,
    # then smaller magnitude.  They never overrule a genuine metric difference.
    tie = (1.0 if alpha == 1.0 else 0.0, 1.0 if alpha == 0.0 else 0.0, -abs(alpha))
    if role == "direction":
        return (value("auc"), value("balanced_accuracy"), value("edge_vs_naive"), *tie)
    if role == "confidence":
        return (
            value("confidence_correctness_auc"), value("hc_accuracy"),
            value("confidence_monotonicity"), value("hc_net"), *tie,
        )
    return (value("material_move_auc"), value("magnitude_spearman"), value("hc_net"), *tie)


def best_alpha(metrics_by_alpha: dict[float, dict[str, Any]], role: str,
               candidates: Iterable[float]) -> float:
    eligible = [float(alpha) for alpha in candidates]
    require(all(alpha in metrics_by_alpha for alpha in eligible), "alpha metrics missing")
    return max(eligible, key=lambda alpha: selection_key(metrics_by_alpha[alpha], role, alpha))


def mode_and_agreement(values: Sequence[float], preferred: float = 0.0) -> tuple[float, float, dict[str, int]]:
    require(bool(values), "cannot compute stability from zero votes")
    counts = Counter(float(value) for value in values)
    maximum = max(counts.values())
    tied = [alpha for alpha, count in counts.items() if count == maximum]
    if preferred in tied:
        chosen = preferred
    elif 1.0 in tied:
        chosen = 1.0
    else:
        chosen = min(tied, key=lambda alpha: (abs(alpha), alpha))
    return float(chosen), float(maximum / len(values)), {
        ALPHA_LABELS[alpha]: int(counts.get(alpha, 0)) for alpha in DOSAGE_ALPHAS
    }


def select_inner_policy(
    fold_metrics: Sequence[dict[float, dict[str, Any]]],
    role: str,
    stability_threshold: float,
    dosage_screen_epsilon: float,
) -> dict[str, Any]:
    """Select alpha from inner metrics only; no outer object is accepted."""
    require(bool(fold_metrics), "inner alpha selection requires fold metrics")
    base_votes = [best_alpha(row, role, BASE_ALPHAS) for row in fold_metrics]
    base_mode, base_agreement, base_counts = mode_and_agreement(base_votes, preferred=0.0)
    plus_vs_zero = []
    for row in fold_metrics:
        plus = finite(row[1.0].get("primary_metric"))
        zero = finite(row[0.0].get("primary_metric"))
        if plus is not None and zero is not None:
            plus_vs_zero.append(plus - zero)
    median_plus_vs_zero = finite(np.median(plus_vs_zero)) if plus_vs_zero else None
    dosage_enabled = bool(
        base_mode != 1.0
        or median_plus_vs_zero is None
        or median_plus_vs_zero < dosage_screen_epsilon
    )
    candidate_grid = DOSAGE_ALPHAS if dosage_enabled else BASE_ALPHAS
    votes = [best_alpha(row, role, candidate_grid) for row in fold_metrics]
    raw_alpha, agreement, counts = mode_and_agreement(votes, preferred=0.0)
    chosen = raw_alpha if agreement >= stability_threshold else 0.0
    return {
        "chosen_alpha": float(chosen), "raw_mode_alpha": float(raw_alpha),
        "stability": agreement, "stable": bool(agreement >= stability_threshold),
        "vote_counts": counts, "base_vote_counts": base_counts,
        "base_mode_alpha": base_mode, "base_agreement": base_agreement,
        "dosage_enabled": dosage_enabled,
        "dosage_screen": {
            "rule": "base_mode_not_plus1_or_median_plus1_minus_zero_below_fixed_epsilon",
            "epsilon": dosage_screen_epsilon,
            "median_plus1_minus_zero_primary": median_plus_vs_zero,
        },
        "candidate_grid": list(candidate_grid),
        "fallback_reason": None if agreement >= stability_threshold else "INNER_STABILITY_BELOW_THRESHOLD",
        "outer_metrics_used": False,
    }


def bounded_source_frame(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    ordered = frame.sort_values(["event_time_utc", "event_id"], kind="mergesort")
    if maximum is None or len(ordered) <= maximum:
        return ordered.reset_index(drop=True)
    # A bounded run deliberately uses a contiguous old-DEV suffix; it does not
    # inspect outcomes to select rows.
    return ordered.iloc[-int(maximum):].reset_index(drop=True)


def evaluate_alpha_grid(
    frame: pd.DataFrame,
    role: str,
    z_full: np.ndarray,
    z_without: np.ndarray,
    material_threshold: float,
) -> dict[float, dict[str, Any]]:
    return {
        alpha: metric_row(
            frame, role, counterfactual_score(z_full, z_without, alpha), material_threshold,
        )
        for alpha in DOSAGE_ALPHAS
    }


def prediction_rows(
    frame: pd.DataFrame,
    source: str,
    block: str,
    role: str,
    outer_fold: int,
    alpha: float,
    z_full: np.ndarray,
    z_without: np.ndarray,
    material_threshold: float,
) -> pd.DataFrame:
    score = counterfactual_score(z_full, z_without, alpha)
    probability = sigmoid(score)
    if role == "direction":
        confidence = np.abs(probability - 0.5) * 2.0
        target = frame.y.to_numpy(int)
        prediction = (probability >= 0.5).astype(int)
        high = top_mask(confidence, int(frame.v69_high_conf.sum()))
        economic_direction_up = prediction
        economic_direction_correct = (prediction == frame.y.to_numpy(int)).astype(int)
    elif role == "confidence":
        confidence = probability
        target = frame.direction_correct.to_numpy(int)
        prediction = (probability >= 0.5).astype(int)
        high = top_mask(confidence, int(frame.v69_high_conf.sum()))
        economic_direction_up = (frame.v69_prob.to_numpy(float) >= 0.5).astype(int)
        economic_direction_correct = frame.direction_correct.to_numpy(int)
    else:
        confidence = pd.Series(score).rank(method="first", pct=True).to_numpy(float)
        target = (np.abs(frame.fwd_ret_30m.to_numpy(float)) >= material_threshold).astype(int)
        prediction = (confidence >= 0.5).astype(int)
        high = top_mask(score, int(frame.v69_opportunity_high.sum()))
        economic_direction_up = (frame.v69_prob.to_numpy(float) >= 0.5).astype(int)
        economic_direction_correct = frame.direction_correct.to_numpy(int)
    return pd.DataFrame({
        "event_id": frame.event_id.to_numpy(), "event_time_utc": frame.event_time_utc.to_numpy(),
        "source_family": source, "feature_block": block, "role": role,
        "outer_fold": outer_fold, "chosen_alpha": alpha,
        "z_full": z_full, "z_without_block": z_without,
        "delta_block": z_full - z_without, "z_selected": score,
        "selected_confidence": confidence, "target": target, "prediction": prediction,
        "correctness": (prediction == target).astype(int), "selected_high": high,
        "direction_correct": frame.direction_correct.to_numpy(int),
        "v69_direction_up": (frame.v69_prob.to_numpy(float) >= 0.5).astype(int),
        "economic_direction_up": economic_direction_up,
        "economic_direction_correct": economic_direction_correct,
        "fwd_ret_30m": frame.fwd_ret_30m.to_numpy(float),
    })


def unsupported_rows(source: str, source_n: int, blocks: Sequence[str]) -> list[dict[str, Any]]:
    return [{
        "source_family": source, "feature_block": block, "role": role,
        "outer_fold": None, "support_status": "INSUFFICIENT_SUPPORT_CONSERVATIVE_ZERO",
        "source_n": source_n, "chosen_alpha": 0.0, "raw_mode_alpha": 0.0,
        "inner_stability": 0.0, "dosage_enabled": False,
        "outer_metrics_used_for_selection": False,
    } for block in blocks for role in ROLE_ORDER]


def run_nested_audit(
    frame: pd.DataFrame,
    registry: dict[str, Any],
    sources: Sequence[str],
    blocks: Sequence[str],
    outer_folds: int,
    inner_folds: int,
    minimum_source_rows: int,
    maximum_rows_per_source: int | None,
    stability_threshold: float,
    dosage_screen_epsilon: float,
    max_text_features: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    audit: dict[str, Any] = {"sources": {}, "outer_labels_used_for_selection": False}
    for source_index, source in enumerate(sources):
        source_all = frame[frame.source_family.eq(source)].copy()
        source_frame = bounded_source_frame(source_all, maximum_rows_per_source)
        outer = expanding_purged_splits(
            source_frame, outer_folds, minimum_train=max(80, minimum_source_rows // 3),
            minimum_valid=max(25, minimum_source_rows // 8),
        )
        supported = len(source_frame) >= minimum_source_rows and bool(outer)
        audit["sources"][source] = {
            "available_rows": len(source_all), "used_rows": len(source_frame),
            "outer_fold_n": len(outer), "minimum_source_rows": minimum_source_rows,
            "supported": supported, "bounded_subset": len(source_frame) < len(source_all),
            "folds": [],
        }
        if not supported:
            fold_rows.extend(unsupported_rows(source, len(source_frame), blocks))
            continue
        for outer_split in outer:
            outer_time = chronology_audit(outer_split)
            inner = expanding_purged_splits(
                outer_split.train, inner_folds,
                minimum_train=max(40, minimum_source_rows // 8), minimum_valid=20,
            )
            require(inner, f"no eligible inner folds: {source} outer={outer_split.fold}")
            inner_metrics: dict[str, dict[str, list[dict[float, dict[str, Any]]]]] = {
                role: {block: [] for block in blocks} for role in ROLE_ORDER
            }
            inner_audits = []
            for inner_split in inner:
                inner_audits.append(chronology_audit(inner_split))
                train_m, valid_m, _ = fit_block_matrices(
                    inner_split.train, inner_split.valid, registry, blocks, max_text_features,
                )
                scores = paired_role_scores(
                    inner_split.train, inner_split.valid, train_m, valid_m, blocks,
                    seed + source_index * 10000 + outer_split.fold * 1000 + inner_split.fold * 10,
                )
                material_threshold = float(np.quantile(
                    np.abs(inner_split.train.fwd_ret_30m.to_numpy(float)), 0.80,
                ))
                for role in ROLE_ORDER:
                    for block in blocks:
                        z_full, z_without = scores[role][block]
                        inner_metrics[role][block].append(evaluate_alpha_grid(
                            inner_split.valid, role, z_full, z_without, material_threshold,
                        ))

            policies = {
                role: {
                    block: select_inner_policy(
                        inner_metrics[role][block], role, stability_threshold,
                        dosage_screen_epsilon,
                    ) for block in blocks
                } for role in ROLE_ORDER
            }
            train_m, valid_m, dimensions = fit_block_matrices(
                outer_split.train, outer_split.valid, registry, blocks, max_text_features,
            )
            outer_scores = paired_role_scores(
                outer_split.train, outer_split.valid, train_m, valid_m, blocks,
                seed + source_index * 10000 + outer_split.fold * 1000 + 500,
            )
            material_threshold = float(np.quantile(
                np.abs(outer_split.train.fwd_ret_30m.to_numpy(float)), 0.80,
            ))
            for role in ROLE_ORDER:
                for block in blocks:
                    policy = policies[role][block]
                    z_full, z_without = outer_scores[role][block]
                    outer_grid = evaluate_alpha_grid(
                        outer_split.valid, role, z_full, z_without, material_threshold,
                    )
                    chosen = float(policy["chosen_alpha"])
                    selected_metrics = outer_grid[chosen]
                    row: dict[str, Any] = {
                        "source_family": source, "feature_block": block, "role": role,
                        "outer_fold": outer_split.fold, "support_status": "SUPPORTED",
                        "source_n": len(source_frame), "outer_train_n": len(outer_split.train),
                        "outer_valid_n": len(outer_split.valid), "inner_fold_n": len(inner),
                        "chosen_alpha": chosen, "raw_mode_alpha": policy["raw_mode_alpha"],
                        "inner_stability": policy["stability"], "inner_stable": policy["stable"],
                        "dosage_enabled": policy["dosage_enabled"],
                        "inner_vote_counts": json.dumps(policy["vote_counts"], sort_keys=True),
                        "inner_fallback_reason": policy["fallback_reason"],
                        "outer_metrics_used_for_selection": False,
                        "raw_feature_sign_flip_performed": False,
                        **{f"selected_{key}": value for key, value in selected_metrics.items()},
                    }
                    for alpha, label in ALPHA_LABELS.items():
                        metrics = outer_grid[alpha]
                        row[label] = metrics["primary_metric"]
                        row[f"{label}_hc_accuracy"] = metrics["hc_accuracy"]
                        row[f"{label}_hc_net"] = metrics["hc_net"]
                    fold_rows.append(row)
                    predictions.append(prediction_rows(
                        outer_split.valid, source, block, role, outer_split.fold,
                        chosen, z_full, z_without, material_threshold,
                    ))
            audit["sources"][source]["folds"].append({
                "outer": outer_time, "inner": inner_audits,
                "feature_dimensions": dimensions,
                "policies_frozen_before_outer": True,
            })
            print(
                f"[V221+ POLARITY] source={source} outer={outer_split.fold} "
                f"inner={len(inner)} blocks={len(blocks)} roles={len(ROLE_ORDER)}",
                flush=True,
            )
    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return fold_frame, prediction_frame, audit


def bootstrap_date_lower(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> float | None:
    selected = frame[frame.selected_high].copy()
    if selected.empty:
        return None
    selected["date"] = pd.to_datetime(selected.event_time_utc, utc=True).dt.date.astype(str)
    dates = np.array(sorted(selected.date.unique()), dtype=object)
    if not len(dates):
        return None
    selected["net"] = (
        np.where(selected.economic_direction_up.to_numpy(int) == 1, 1.0, -1.0)
        * selected.fwd_ret_30m.to_numpy(float) - COST
    )
    grouped = {date: group.net.to_numpy(float) for date, group in selected.groupby("date")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        values = np.concatenate([grouped[str(date)] for date in sample_dates])
        draws.append(float(values.mean()))
    return float(np.quantile(draws, 0.025))


def economic_stats(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame[frame.selected_high].copy()
    if selected.empty:
        return {
            "hc_net": None, "winsorized_hc_net": None,
            "top1_removed_hc_net": None, "top5_removed_hc_net": None,
        }
    net = (
        np.where(selected.economic_direction_up.to_numpy(int) == 1, 1.0, -1.0)
        * selected.fwd_ret_30m.to_numpy(float) - COST
    )
    low, high = np.quantile(net, [0.01, 0.99]) if len(net) >= 5 else (net.min(), net.max())
    order = np.argsort(np.abs(net))[::-1]

    def removed(count: int) -> float | None:
        keep = np.ones(len(net), dtype=bool)
        keep[order[:min(count, len(net))]] = False
        return float(net[keep].mean()) if keep.any() else None

    return {
        "hc_net": float(net.mean()),
        "winsorized_hc_net": float(np.clip(net, low, high).mean()),
        "top1_removed_hc_net": removed(1), "top5_removed_hc_net": removed(5),
    }


def extreme_contribution_report(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, str], bool]]:
    columns = [
        "source_family", "feature_block", "role", "contribution_quantile_band",
        "n", "mean_abs_contribution", "accuracy", "auc", "correctness",
        "mean_confidence", "hc_accuracy", "net_return", "extreme_contribution_reversal",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns), {}
    rows: list[dict[str, Any]] = []
    flags: dict[tuple[str, str, str], bool] = {}
    for key, group in predictions.groupby(["source_family", "feature_block", "role"], sort=True):
        work = group.copy().reset_index(drop=True)
        work["abs_delta"] = np.abs(work.delta_block.to_numpy(float))
        work["magnitude_pct"] = work.abs_delta.rank(method="first", pct=True)
        group_rows = []
        for low, high, label in EXTREME_BANDS:
            mask = (work.magnitude_pct > low) & (work.magnitude_pct <= high)
            subset = work[mask]
            target = subset.target.to_numpy(int)
            score = subset.z_selected.to_numpy(float)
            high_subset = subset[subset.selected_high]
            row = {
                "source_family": key[0], "feature_block": key[1], "role": key[2],
                "contribution_quantile_band": label, "n": len(subset),
                "mean_abs_contribution": float(subset.abs_delta.mean()) if len(subset) else None,
                "accuracy": float(subset.correctness.mean()) if len(subset) else None,
                "auc": safe_auc(target, score) if len(subset) else None,
                "correctness": float(subset.correctness.mean()) if len(subset) else None,
                "mean_confidence": float(subset.selected_confidence.mean()) if len(subset) else None,
                "hc_accuracy": (
                    float(high_subset.economic_direction_correct.mean()) if len(high_subset) else None
                ),
                "net_return": (
                    float((np.where(subset.economic_direction_up.to_numpy(int) == 1, 1.0, -1.0)
                           * subset.fwd_ret_30m.to_numpy(float) - COST).mean())
                    if len(subset) else None
                ),
            }
            group_rows.append(row)
        low_row, high_row = group_rows[0], group_rows[-1]
        reversal = bool(
            low_row["correctness"] is not None and high_row["correctness"] is not None
            and low_row["mean_confidence"] is not None and high_row["mean_confidence"] is not None
            and high_row["mean_confidence"] > low_row["mean_confidence"]
            and high_row["correctness"] < low_row["correctness"]
        )
        flags[key] = reversal
        for row in group_rows:
            row["extreme_contribution_reversal"] = reversal
            rows.append(row)
    return pd.DataFrame(rows, columns=columns), flags


def classify_reversal(
    chosen_alpha: float,
    stability: float,
    alpha_scores: dict[float, float | None],
    stability_threshold: float,
    extreme_reversal: bool,
    epsilon: float = 0.002,
) -> str:
    flags: list[str] = []
    if stability < stability_threshold:
        flags.append("UNSTABLE")
    else:
        plus = finite(alpha_scores.get(1.0))
        zero = finite(alpha_scores.get(0.0))
        plus05 = finite(alpha_scores.get(0.5))
        plus15 = finite(alpha_scores.get(1.5))
        if chosen_alpha == -1.0:
            flags.append("POLARITY_REVERSED")
        elif chosen_alpha == 0.0:
            flags.append(
                "HARMFUL" if plus is not None and zero is not None and zero - plus > epsilon
                else "REDUNDANT"
            )
        elif chosen_alpha == 0.5:
            if (plus05 is not None and plus is not None and plus15 is not None
                    and plus05 > plus and plus15 < plus):
                flags.append("OVERWEIGHTED")
            else:
                flags.append("SATURATING")
        else:
            flags.append("SUPPORTIVE")
            if (plus05 is not None and plus is not None and plus15 is not None
                    and plus05 >= plus - epsilon and plus15 < plus - epsilon):
                flags.append("SATURATING")
    if extreme_reversal:
        flags.append("EXTREME_REVERSAL")
    return "|".join(dict.fromkeys(flags))


def aggregate_scorecards(
    fold_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    sources: Sequence[str],
    blocks: Sequence[str],
    stability_threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    extreme, extreme_flags = extreme_contribution_report(predictions)
    scorecards: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {}
    for source in sources:
        mapping[source] = {}
        for block in blocks:
            mapping[source][block] = {}
            for role in ROLE_ORDER:
                group = fold_rows[
                    fold_rows.source_family.eq(source)
                    & fold_rows.feature_block.eq(block)
                    & fold_rows.role.eq(role)
                    & fold_rows.support_status.eq("SUPPORTED")
                ].copy()
                if group.empty:
                    source_rows = fold_rows[
                        fold_rows.source_family.eq(source)
                        & fold_rows.feature_block.eq(block)
                        & fold_rows.role.eq(role)
                    ]
                    available_support = (
                        int(source_rows.source_n.max()) if not source_rows.empty else 0
                    )
                    score = {
                        "source_family": source, "feature_block": block, "role": role,
                        "support_status": "INSUFFICIENT_SUPPORT_CONSERVATIVE_ZERO",
                        "support": available_support, "evaluated_support": 0,
                        "chosen_alpha": 0.0, "stability": 0.0,
                        "AUC": None, "BA": None, "edge": None,
                        "median_fold_AUC": None, "worst_fold_AUC": None,
                        "AUC_fold_consistency": None,
                        "confidence_correctness_auc": None, "HC_accuracy": None,
                        "HC_net": None, "bootstrap_HC_lower": None,
                        "winsorized_HC_net": None, "top1_removed_HC_net": None,
                        "top5_removed_HC_net": None, "source_specific_net": None,
                        "material_move_AUC": None, "magnitude_ranking": None,
                        "outer_metrics_used_for_selection": False,
                        "raw_feature_sign_flip_performed": False,
                    }
                    score.update({label: None for label in ALPHA_LABELS.values()})
                    scorecards.append(score)
                    classification = "UNSTABLE"
                    classifications.append({
                        "source_family": source, "feature_block": block, "role": role,
                        "classification": classification, "chosen_alpha": 0.0,
                        "stability": 0.0, "support": available_support,
                        "extreme_contribution_reversal": False,
                    })
                    stability_rows.append({
                        "source_family": source, "feature_block": block, "role": role,
                        "chosen_alpha": 0.0, "stability": 0.0, "outer_policy_fold_n": 0,
                        "stable": False, "vote_counts": json.dumps({"alpha_zero": 0}),
                        "fallback_reason": "INSUFFICIENT_SUPPORT_CONSERVATIVE_ZERO",
                    })
                    mapping[source][block][role] = {
                        "alpha": 0.0, "stability": 0.0, "support": available_support,
                        "status": "NO_STABLE_POLARITY_SIGNAL",
                    }
                    continue

                outer_votes = group.chosen_alpha.astype(float).tolist()
                mode_alpha, agreement, counts = mode_and_agreement(outer_votes, preferred=0.0)
                stable = agreement >= stability_threshold
                recommended = mode_alpha if stable else 0.0
                prediction_group = predictions[
                    predictions.source_family.eq(source)
                    & predictions.feature_block.eq(block)
                    & predictions.role.eq(role)
                ].copy()
                economic = economic_stats(prediction_group)
                lower = bootstrap_date_lower(
                    prediction_group, bootstrap_samples,
                    seed + int.from_bytes(
                        hashlib.sha256(f"{source}|{block}|{role}".encode("utf-8")).digest()[:4],
                        "big",
                    ) % 100000,
                )
                row: dict[str, Any] = {
                    "source_family": source, "feature_block": block, "role": role,
                    "support_status": "SUPPORTED", "support": int(len(prediction_group)),
                    "outer_fold_n": int(len(group)), "chosen_alpha": recommended,
                    "nested_applied_alpha_mode": mode_alpha, "stability": agreement,
                    "AUC": finite(group.selected_auc.mean()),
                    "BA": finite(group.selected_balanced_accuracy.mean()),
                    "edge": finite(group.selected_edge_vs_naive.mean()),
                    "median_fold_AUC": finite(group.selected_auc.median()),
                    "worst_fold_AUC": finite(group.selected_auc.min()),
                    "AUC_fold_consistency": finite(group.selected_auc.std(ddof=0)),
                    "confidence_correctness_auc": finite(
                        group.selected_confidence_correctness_auc.mean()
                    ),
                    "HC_accuracy": finite(group.selected_hc_accuracy.mean()),
                    "HC_net": economic["hc_net"], "bootstrap_HC_lower": lower,
                    "source_specific_net": economic["hc_net"],
                    "winsorized_HC_net": economic["winsorized_hc_net"],
                    "top1_removed_HC_net": economic["top1_removed_hc_net"],
                    "top5_removed_HC_net": economic["top5_removed_hc_net"],
                    "material_move_AUC": finite(group.selected_material_move_auc.mean()),
                    "magnitude_ranking": finite(group.selected_magnitude_spearman.mean()),
                    "dosage_evaluated_outer_fold_fraction": float(group.dosage_enabled.mean()),
                    "outer_metrics_used_for_selection": False,
                    "raw_feature_sign_flip_performed": False,
                }
                alpha_scores: dict[float, float | None] = {}
                for alpha, label in ALPHA_LABELS.items():
                    value = finite(group[label].mean())
                    row[label] = value
                    alpha_scores[alpha] = value
                chosen_outer_primary = finite(alpha_scores.get(recommended))
                basic_controls = [
                    finite(alpha_scores.get(alpha)) for alpha in BASE_ALPHAS
                    if alpha != recommended
                ]
                basic_controls = [value for value in basic_controls if value is not None]
                best_basic_control = max(basic_controls) if basic_controls else None
                outer_margin = (
                    chosen_outer_primary - best_basic_control
                    if chosen_outer_primary is not None and best_basic_control is not None
                    else None
                )
                # Outer labels evaluate the already-frozen inner policy only.
                # This flag is evidence, never a reselection instruction.
                outer_state_confirmed = bool(
                    outer_margin is not None and outer_margin > 0.002
                )
                row.update({
                    "outer_fixed_state_primary": chosen_outer_primary,
                    "outer_best_basic_control_primary": best_basic_control,
                    "outer_fixed_state_margin": outer_margin,
                    "outer_fixed_state_confirmed": outer_state_confirmed,
                    "new_data_replication": False,
                    "production_material_candidate": False,
                })
                extreme_reversal = extreme_flags.get((source, block, role), False)
                classification = classify_reversal(
                    recommended, agreement, alpha_scores, stability_threshold, extreme_reversal,
                )
                if (agreement >= stability_threshold and recommended != 0.0
                        and not outer_state_confirmed):
                    classification += "|OUTER_NOT_CONFIRMED"
                scorecards.append(row)
                classifications.append({
                    "source_family": source, "feature_block": block, "role": role,
                    "classification": classification, "chosen_alpha": recommended,
                    "nested_applied_alpha_mode": mode_alpha, "stability": agreement,
                    "support": len(prediction_group),
                    "extreme_contribution_reversal": extreme_reversal,
                    "outer_fixed_state_primary": chosen_outer_primary,
                    "outer_best_basic_control_primary": best_basic_control,
                    "outer_fixed_state_margin": outer_margin,
                    "outer_fixed_state_confirmed": outer_state_confirmed,
                    "new_data_replication": False,
                    "production_material_candidate": False,
                    "block_level_authority": True,
                    "individual_feature_drilldown_authorized": bool(
                        stable and (recommended in {-1.0, 0.0, 0.5} or extreme_reversal)
                    ),
                })
                stability_rows.append({
                    "source_family": source, "feature_block": block, "role": role,
                    "chosen_alpha": recommended, "nested_applied_alpha_mode": mode_alpha,
                    "stability": agreement, "outer_policy_fold_n": len(outer_votes),
                    "stable": stable, "vote_counts": json.dumps(counts, sort_keys=True),
                    "fallback_reason": None if stable else "OUTER_INNER_POLICY_AGREEMENT_BELOW_THRESHOLD",
                    "selection_evidence": "INNER_FOLDS_ONLY",
                })
                mapping[source][block][role] = {
                    "alpha": recommended, "nested_applied_alpha_mode": mode_alpha,
                    "stability": agreement, "support": len(prediction_group),
                    "classification": classification,
                    "outer_fixed_state_margin": outer_margin,
                    "outer_fixed_state_confirmed": outer_state_confirmed,
                    "new_data_replication": False,
                    "status": (
                        "OLD_DEV_OUTER_CONFIRMED_DIAGNOSTIC_ONLY"
                        if outer_state_confirmed
                        else "OLD_DEV_INNER_SIGNAL_OUTER_NOT_CONFIRMED"
                    ),
                }
    return (
        pd.DataFrame(scorecards), pd.DataFrame(classifications),
        pd.DataFrame(stability_rows), {"source_polarity_map": mapping, "extreme": extreme},
    )


def reliability_metrics(frame: pd.DataFrame, score_column: str, target_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"n": 0}
    work = frame.copy().sort_values([score_column, "event_time_utc"], kind="mergesort").reset_index(drop=True)
    score = work[score_column].to_numpy(float)
    target = work[target_column].to_numpy(int)
    work["bin"] = np.minimum(9, np.arange(len(work)) * 10 // len(work)) + 1
    bins = [{
        "bin": int(number), "n": len(group),
        "mean_confidence": float(group[score_column].mean()),
        "accuracy": float(group[target_column].mean()),
        "net": float((
            np.where(group.v69_direction_up.to_numpy(int) == 1, 1.0, -1.0)
            * group.fwd_ret_30m.to_numpy(float) - COST
        ).mean()),
    } for number, group in work.groupby("bin", sort=True)]
    order = np.argsort(-score, kind="stable")

    def top(coverage: float) -> dict[str, Any]:
        count = max(1, int(math.ceil(len(work) * coverage)))
        chosen = work.iloc[order[:count]]
        return {
            "coverage": coverage, "n": count,
            "accuracy": float(chosen[target_column].mean()),
            "net": float((
                np.where(chosen.v69_direction_up.to_numpy(int) == 1, 1.0, -1.0)
                * chosen.fwd_ret_30m.to_numpy(float) - COST
            ).mean()),
        }

    bin_accuracy = pd.Series([row["accuracy"] for row in bins], dtype=float)
    return {
        "n": len(work), "confidence_correctness_auc": safe_auc(target, score),
        "top_10_percent": top(0.10), "top_20_percent": top(0.20),
        "confidence_bins": bins,
        "confidence_bin_monotonicity": finite(
            pd.Series(np.arange(len(bins)), dtype=float).corr(bin_accuracy, method="spearman")
        ),
        "risk_coverage": [top(coverage) for coverage in (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)],
    }


def confidence_reliability_report(
    input_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    blocks: Sequence[str],
) -> dict[str, Any]:
    baseline = input_frame[[
        "event_id", "event_time_utc", "source_family", "direction_correct",
        "v69_prob", "v69_confidence_signal", "fwd_ret_30m",
    ]].copy()
    baseline["v69_direction_up"] = (baseline.v69_prob >= 0.5).astype(int)
    scopes = ["OVERALL", *SOURCE_ORDER]
    report: dict[str, Any] = {
        "version": VERSION, "diagnostic_only": True,
        "baseline_v69": {}, "selected_counterfactual_by_block": {},
    }
    for scope in scopes:
        subset = baseline if scope == "OVERALL" else baseline[baseline.source_family.eq(scope)]
        if not subset.empty:
            report["baseline_v69"][scope] = reliability_metrics(
                subset, "v69_confidence_signal", "direction_correct",
            )
    confidence = (
        predictions[predictions.role.eq("confidence")].copy()
        if not predictions.empty and "role" in predictions.columns
        else pd.DataFrame()
    )
    for block in blocks:
        report["selected_counterfactual_by_block"][block] = {}
        block_frame = (
            confidence[confidence.feature_block.eq(block)].copy()
            if not confidence.empty else pd.DataFrame()
        )
        if block_frame.empty:
            continue
        for scope in scopes:
            subset = block_frame if scope == "OVERALL" else block_frame[
                block_frame.source_family.eq(scope)
            ]
            if not subset.empty:
                report["selected_counterfactual_by_block"][block][scope] = reliability_metrics(
                    subset, "selected_confidence", "target",
                )
    return report


def resolve_output_dir(value: str | None, smoke: bool) -> Path:
    target = Path(value).resolve() if value else (
        DEFAULT_AUDIT_ROOT / ("SMOKE" if smoke else "FULL_OLD_DEV")
    ).resolve()
    allowed = [(ROOT / "research" / "audits").resolve(), (ROOT / "staging").resolve()]
    require(any(target == root or root in target.parents for root in allowed),
            "output directory must be under research/audits or staging")
    return target


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def comma_values(value: str | None, available: Sequence[str]) -> list[str]:
    if not value:
        return list(available)
    selected = [piece.strip() for piece in value.split(",") if piece.strip()]
    require(bool(selected), "selection cannot be empty")
    unknown = set(selected) - set(available)
    require(not unknown, f"unknown selections: {sorted(unknown)}")
    return selected


def write_reports(
    output_dir: Path,
    scorecard: pd.DataFrame,
    classification: pd.DataFrame,
    stability: pd.DataFrame,
    extreme: pd.DataFrame,
    fold_rows: pd.DataFrame,
    polarity_map: dict[str, Any],
    confidence_report: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, str]:
    tables = {
        "POLARITY_REVERSAL_SCORECARD_V221PLUS.csv": scorecard,
        "FEATURE_REVERSAL_CLASSIFICATION_V221PLUS.csv": classification,
        "POLARITY_STABILITY_V221PLUS.csv": stability,
        "EXTREME_CONTRIBUTION_DIAGNOSTIC_V221PLUS.csv": extreme,
        "NESTED_OUTER_FOLD_AUDIT_V221PLUS.csv": fold_rows,
    }
    payloads = {
        "FEATURE_CONTRIBUTION_POLARITY_MAP_V221PLUS.json": polarity_map,
        "CONFIDENCE_RELIABILITY_REPORT_V221PLUS.json": confidence_report,
    }
    for name, frame in tables.items():
        atomic_csv(frame, output_dir / name)
    for name, payload in payloads.items():
        atomic_json(payload, output_dir / name)
    hashes = {
        name: sha256(output_dir / name) for name in sorted([*tables, *payloads])
    }
    manifest = {**manifest, "artifact_sha256": hashes, "artifacts": sorted(hashes)}
    atomic_json(manifest, output_dir / "MANIFEST.json")
    return {**hashes, "MANIFEST.json": sha256(output_dir / "MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true", help="verify inputs and write nothing")
    mode.add_argument("--smoke-test", action="store_true", help="bounded US_NEWS/two-block audit")
    mode.add_argument("--full-old-dev", action="store_true", help="run all requested old-DEV rows")
    parser.add_argument("--sources", help="comma-separated source families")
    parser.add_argument("--blocks", help="comma-separated registry block names")
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--minimum-source-rows", type=int, default=200)
    parser.add_argument("--max-rows-per-source", type=int)
    parser.add_argument("--max-text-features", type=int, default=6000)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--stability-threshold", type=float, default=0.75)
    parser.add_argument("--dosage-screen-epsilon", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=22101)
    parser.add_argument("--cpu-ids", default="30,31")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = configure_runtime(parse_cpu_ids(args.cpu_ids))
    registry, registry_hash = load_registry()
    frame, input_audit = verify_and_load_inputs(registry)
    available_blocks = list(registry["blocks"])
    sources = comma_values(args.sources, SOURCE_ORDER)
    blocks = comma_values(args.blocks, available_blocks)
    if args.audit_only:
        print(json.dumps(json_clean({
            "status": "AUDIT_OK", "version": VERSION, "runtime": runtime,
            "registry_sha256": registry_hash, "input_audit": input_audit,
            "selected_sources": sources, "selected_blocks": blocks,
            "output_written": False, "raw_feature_sign_flip_performed": False,
        }), ensure_ascii=False, indent=2))
        return

    outer_folds = args.outer_folds
    inner_folds = args.inner_folds
    maximum_rows = args.max_rows_per_source
    max_text_features = args.max_text_features
    bootstrap_samples = args.bootstrap_samples
    minimum_source_rows = args.minimum_source_rows
    if args.smoke_test:
        sources = ["US_NEWS"] if not args.sources else sources[:1]
        blocks = blocks[:2]
        outer_folds = min(2, outer_folds)
        inner_folds = min(2, inner_folds)
        maximum_rows = min(maximum_rows or 600, 600)
        max_text_features = min(max_text_features, 2000)
        bootstrap_samples = min(bootstrap_samples, 50)
        minimum_source_rows = min(minimum_source_rows, 200)
    require(0.70 <= args.stability_threshold <= 0.75,
            "stability threshold must remain in the predeclared 70-75% range")
    require(outer_folds >= 1 and inner_folds >= 1, "fold counts must be positive")
    require(minimum_source_rows >= 40, "minimum source support must be at least 40")
    require(bootstrap_samples >= 20, "bootstrap sample count must be at least 20")

    fold_rows, predictions, nested_audit = run_nested_audit(
        frame, registry, sources, blocks, outer_folds, inner_folds,
        minimum_source_rows, maximum_rows, args.stability_threshold,
        args.dosage_screen_epsilon, max_text_features, args.seed,
    )
    scorecard, classification, stability, aggregate = aggregate_scorecards(
        fold_rows, predictions, sources, blocks, args.stability_threshold,
        bootstrap_samples, args.seed,
    )
    extreme = aggregate.pop("extreme")
    entries = []
    for source, source_map in aggregate["source_polarity_map"].items():
        for block, block_map in source_map.items():
            entries.append({
                "source_family": source, "feature_block": block,
                "direction_alpha": block_map["direction"]["alpha"],
                "confidence_alpha": block_map["confidence"]["alpha"],
                "opportunity_alpha": block_map["opportunity"]["alpha"],
                "stability": {
                    role: block_map[role]["stability"] for role in ROLE_ORDER
                },
                "support": {role: block_map[role]["support"] for role in ROLE_ORDER},
                "production_authorized": False,
            })
    polarity_map = {
        "version": VERSION, "diagnostic_only": True,
        "selection_scope": "INNER_PAST_ONLY", "outer_labels_for_selection": False,
        "raw_feature_sign_flip_allowed": False,
        "raw_feature_sign_flip_performed": False,
        "limited_dosage_grid": list(DOSAGE_ALPHAS),
        "stability_threshold": args.stability_threshold,
        "entries": entries, **aggregate,
    }
    confidence_report = confidence_reliability_report(frame, predictions, blocks)
    output_dir = resolve_output_dir(args.output_dir, args.smoke_test)
    manifest = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "mode": "SMOKE" if args.smoke_test else "FULL_OLD_DEV",
        "output_directory": str(output_dir), "runtime": runtime,
        "implementation": {
            "path": Path(__file__).resolve().name,
            "sha256": sha256(Path(__file__).resolve()),
            "test_path": "test_v221plus_contribution_polarity_diagnostic.py",
            "test_sha256": sha256(ROOT / "test_v221plus_contribution_polarity_diagnostic.py"),
            "predecessor_sha256": {
                "experiment_v64_feature_polarity_diagnostic.py": sha256(
                    ROOT / "experiment_v64_feature_polarity_diagnostic.py"
                ),
                "experiment_v67_direction_opportunity_polarity_diagnostic.py": sha256(
                    ROOT / "experiment_v67_direction_opportunity_polarity_diagnostic.py"
                ),
            },
        },
        "registry_path": str(REGISTRY_PATH.relative_to(ROOT)),
        "registry_sha256": registry_hash, "input_audit": input_audit,
        "configuration": {
            "sources": sources, "blocks": blocks, "outer_folds": outer_folds,
            "inner_folds": inner_folds, "minimum_source_rows": minimum_source_rows,
            "maximum_rows_per_source": maximum_rows,
            "max_text_features": max_text_features,
            "bootstrap_samples": bootstrap_samples,
            "stability_threshold": args.stability_threshold,
            "dosage_screen_epsilon": args.dosage_screen_epsilon, "seed": args.seed,
        },
        "nested_chronology_audit": nested_audit,
        "invariants": {
            "outer_labels_used_for_selection": False,
            "policies_frozen_before_outer": True,
            "raw_feature_sign_flip_performed": False,
            "seal_or_reserve_opened": False,
            "registries_or_committed_outputs_modified": False,
            "production_intervention_authorized": False,
        },
    }
    hashes = write_reports(
        output_dir, scorecard, classification, stability, extreme, fold_rows,
        polarity_map, confidence_report, manifest,
    )
    print(json.dumps(json_clean({
        "status": "SMOKE_OK" if args.smoke_test else "FULL_OLD_DEV_DIAGNOSTIC_WRITTEN",
        "version": VERSION, "output_dir": str(output_dir),
        "scorecard_rows": len(scorecard), "classification_rows": len(classification),
        "extreme_rows": len(extreme), "artifact_sha256": hashes,
        "outer_labels_used_for_selection": False,
        "raw_feature_sign_flip_performed": False,
        "production_intervention_authorized": False,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
