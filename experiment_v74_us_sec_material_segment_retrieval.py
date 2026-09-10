"""V74 material US_SEC direction challenger: focused segments + past retrieval.

This standalone draft runner is intentionally disjoint from V71 causal target
priors and V57 full-document transformer adaptation.  It reads only the pinned
immutable V36 DEV table and the committed V69 champion artifacts.  SEC text is
reduced, before vectorization, to filing headline/form/items, selected material
sentences, and selected EX-99 sentences; a full filing is never a model input.

Outer folds 2--4 are chronological.  Every outer and inner training cut ends
strictly more than 35 minutes before its validation cut.  Inner-past OOF labels
alone choose one of four predeclared coarse architectures or exact V69 no-op.
Outer outcomes are evaluation-only.  A failed material gate returns exact V69
probability, confidence_signal, and high_conf arrays.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

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
        str(ROOT / "staging" / "V74_US_SEC_MATERIAL_SEGMENT_RETRIEVAL_V1"),
    )
)

VERSION = 74
HYPOTHESIS = "US_SEC_MATERIAL_SEGMENT_RETRIEVAL_V1"
EXPECTED_SHA256 = {
    DEV_PATH: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V69_PATH: "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002",
}
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V69_COMMIT_SHA256 = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA256 = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EXPECTED_V69_ROBUSTNESS_SHA256 = "778c5a87e4be328d702f85eb7298bcbb70e8b54d3429b19a0dc6e2c7e156cce1"
EXPECTED_DEV_ROWS = 9462
EXPECTED_SEC_TRAIN_ROWS = 7022
EXPECTED_OOF_ROWS = 7568
EXPECTED_US_SEC_OOF_ROWS = 5426

EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
CLASSIFIER_C = 0.50
RETRIEVAL_NEIGHBORS = 24
RETRIEVAL_TEMPERATURE = 0.10
RETRIEVAL_PRIOR_STRENGTH = 8.0
SEED = 7401
BOOTSTRAP_DRAWS = 2000

ITEM_RE = re.compile(r"(?i)\bitem\s+(1\.01|2\.02|2\.03|3\.02|5\.02|7\.01|8\.01|9\.01)\b")
EX99_RE = re.compile(r"(?mi)^\s*(?:EX(?:HIBIT)?[-_ ]?99(?:\.\d+)?)\s*$")
SIGNATURE_RE = re.compile(r"(?mi)^\s*SIGNATURES?\s*$")
DISCLAIMER_RE = re.compile(r"(?is)(?:forward[- ]looking statements?|safe harbor).{0,12000}$")
UNIT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{1,}")
SPACE_RE = re.compile(r"\s+")

# Fixed, label-independent material cues.  They define representation, not a
# hand-authored direction rule; no polarity or sign is attached to any cue.
MATERIAL_CUES = (
    "fda", "approval", "complete response letter", "clinical hold", "phase 1",
    "phase 2", "phase 3", "primary endpoint", "secondary endpoint", "topline",
    "statistically significant", "adverse event", "safety", "recall",
    "public offering", "registered direct", "at-the-market", "convertible",
    "warrant", "priced", "proceeds", "licensing agreement", "collaboration",
    "upfront payment", "milestone", "merger", "acquisition", "guidance",
    "results of operations", "revenue", "net income", "cash and cash equivalents",
    "appoint", "resign", "manufacturing", "deficiency", "material agreement",
)
NUMERIC_CUE_RE = re.compile(
    r"(?i)(?:\$\s*\d|\b\d+(?:\.\d+)?\s*%|\bp\s*[=<]\s*0?\.\d+|hazard ratio|\bn\s*=\s*\d+)"
)

ARCHITECTURES = (
    {
        "name": "HEADLINE_FORM_ITEM_LOGREG_BLEND",
        "weights": {"v69": 0.50, "headline_form_item": 0.50},
    },
    {
        "name": "MATERIAL_EX99_LOGREG_BLEND",
        "weights": {"v69": 0.50, "material_ex99": 0.50},
    },
    {
        "name": "FOCUSED_PAST_RETRIEVAL_BLEND",
        "weights": {"v69": 0.50, "retrieval": 0.50},
    },
    {
        "name": "MATERIAL_RETRIEVAL_CONSENSUS",
        "weights": {"v69": 0.50, "material_ex99": 0.25, "retrieval": 0.25},
    },
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


def metric(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
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


def verify_and_load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"required V74 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V74 input changed: {path.name}")
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

    data = pd.read_csv(DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    champion = pd.read_csv(V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    require(len(data) == EXPECTED_DEV_ROWS, "immutable DEV row count changed")
    require(len(champion) == EXPECTED_OOF_ROWS, "V69 OOF row count changed")
    require(data.event_id.is_unique and champion.event_id.is_unique, "event_id uniqueness failed")
    require(set(champion.event_id).issubset(set(data.event_id)), "V69 event outside immutable DEV")
    require(int(champion.source_family.eq("US_SEC").sum()) == EXPECTED_US_SEC_OOF_ROWS, "US_SEC OOF changed")
    data["event_time_utc"] = pd.to_datetime(data.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    data["headline"] = data.headline.fillna("").astype(str)
    data["body"] = data.body.fillna("").astype(str)
    sec = data.loc[data.source_family.eq("US_SEC")].copy()
    require(len(sec) == EXPECTED_SEC_TRAIN_ROWS and sec.event_id.is_unique, "SEC DEV universe changed")
    require(sec.y.isin([0, 1]).all(), "non-binary direction target")
    require(sec.headline.str.strip().ne("").all(), "US_SEC headline contract changed")
    audit = {
        "input_sha256": hashes,
        "authorized_data_inputs": [
            str(DEV_PATH.relative_to(ROOT)),
            str(V69_PATH.relative_to(ROOT)),
        ],
        "authorized_v69_integrity_inputs": [
            str(V69_COMMIT_PATH.relative_to(ROOT)),
            str(V69_MANIFEST_PATH.relative_to(ROOT)),
            str(V69_ROBUSTNESS_PATH.relative_to(ROOT)),
        ],
        "immutable_dev_only": True,
        "extension_data_opened": False,
        "assignment_file_opened": False,
        "reserved_evaluation_opened": False,
        "sec_cache_recovery_performed": False,
        "dev_rows": len(data),
        "sec_training_rows": len(sec),
        "sec_body_present_rows": int(sec.body.str.strip().ne("").sum()),
        "v69_oof_rows": len(champion),
        "v69_us_sec_oof_rows": int(champion.source_family.eq("US_SEC").sum()),
        "v69_atomic_commit_verified": True,
    }
    return sec, champion, audit


def normalize_unit(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "")).strip()


def material_score(unit: str) -> int:
    lower = unit.lower()
    score = 0
    score += 3 * sum(cue in lower for cue in MATERIAL_CUES)
    score += 2 * int(bool(NUMERIC_CUE_RE.search(unit)))
    score += 2 * int(bool(ITEM_RE.search(unit)))
    score += int(40 <= len(unit) <= 700)
    return score


def select_units(text: str, limit: int, max_chars: int) -> str:
    units = [normalize_unit(unit) for unit in UNIT_SPLIT_RE.split(text) if normalize_unit(unit)]
    ranked = sorted(
        ((material_score(unit), position, unit[:900]) for position, unit in enumerate(units)),
        key=lambda row: (-row[0], row[1]),
    )
    positive = [row for row in ranked if row[0] > 0][:limit]
    if not positive:
        positive = [(0, position, unit[:900]) for position, unit in enumerate(units[: min(3, limit)])]
    chosen = sorted(positive, key=lambda row: row[1])
    return normalize_unit(" ".join(row[2] for row in chosen))[:max_chars]


def focused_segments(row: pd.Series) -> dict[str, Any]:
    headline = normalize_unit(row.get("headline", ""))[:600]
    form = normalize_unit(row.get("form", "UNKNOWN")).upper()[:40] or "UNKNOWN"
    raw_body = str(row.get("body", "") or "")
    body = DISCLAIMER_RE.sub(" ", raw_body)
    signature = SIGNATURE_RE.search(body)
    if signature:
        body = body[: signature.start()]
    items = sorted(set(ITEM_RE.findall(f"{headline}\n{body}")))
    item_text = " ".join(f"ITEM {item}" for item in items) or "ITEM NONE"
    header = normalize_unit(f"FORM {form} {item_text} HEADLINE {headline}")[:1000]
    ex99_match = EX99_RE.search(body)
    main_body = body[: ex99_match.start()] if ex99_match else body
    ex99_body = body[ex99_match.end() :] if ex99_match else ""
    material = select_units(main_body, limit=12, max_chars=4000)
    ex99 = select_units(ex99_body, limit=7, max_chars=2200) if ex99_body else ""
    material_ex99 = normalize_unit(f"{header} MATERIAL {material} EX99 {ex99}")[:7200]
    return {
        "headline_form_item_text": header,
        "material_ex99_text": material_ex99,
        "body_present": bool(raw_body.strip()),
        "items": tuple(items),
        "item_present": bool(items),
        "ex99_present": bool(ex99_body.strip()),
        "material_present": bool(material.strip()),
        "segment_chars": len(material_ex99),
        "raw_body_chars": len(raw_body),
    }


def prepare_sec(sec: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = sec.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True).copy()
    segments = [focused_segments(row) for _, row in work.iterrows()]
    for key in (
        "headline_form_item_text",
        "material_ex99_text",
        "body_present",
        "items",
        "item_present",
        "ex99_present",
        "material_present",
        "segment_chars",
        "raw_body_chars",
    ):
        work[key] = [value[key] for value in segments]
    work["segment_cluster_id"] = [
        hashlib.sha256(normalize_unit(text).lower().encode("utf-8")).hexdigest()[:20]
        for text in work.material_ex99_text
    ]
    audit = {
        "representation": "headline/form/item + selected material sentences + selected EX-99 sentences",
        "full_document_is_model_input": False,
        "full_document_transformer_used": False,
        "max_material_sentences": 12,
        "max_ex99_sentences": 7,
        "max_model_input_chars": 7200,
        "observed_max_model_input_chars": int(work.segment_chars.max()),
        "rows": len(work),
        "body_present_rows": int(work.body_present.sum()),
        "item_present_rows": int(work.item_present.sum()),
        "ex99_present_rows": int(work.ex99_present.sum()),
        "material_present_rows": int(work.material_present.sum()),
        "median_raw_body_chars": float(work.raw_body_chars.median()),
        "median_segment_chars": float(work.segment_chars.median()),
        "segment_cluster_count": int(work.segment_cluster_id.nunique()),
        "selection_is_label_independent": True,
    }
    require(audit["observed_max_model_input_chars"] <= 7200, "focused segment size contract failed")
    return work, audit


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
    sec: pd.DataFrame,
    champion_sec: pd.DataFrame,
    outer_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible = champion_sec.loc[champion_sec.event_time_utc < outer_start - EMBARGO].copy()
    eligible = eligible.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(eligible) >= 300, "insufficient prior OOF evidence")
    boundary = max(100, int(math.floor(len(eligible) * (1.0 - INNER_VALID_FRACTION))))
    inner_valid_ids = set(eligible.iloc[boundary:].event_id)
    inner_valid = sec.loc[sec.event_id.isin(inner_valid_ids)].copy()
    inner_valid = inner_valid.sort_values(["event_time_utc", "event_id"], kind="stable")
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = sec.loc[sec.event_time_utc < inner_start - EMBARGO].copy()
    valid_groups = set(inner_valid.event_group_id.astype(str))
    inner_train = inner_train.loc[~inner_train.event_group_id.astype(str).isin(valid_groups)].copy()
    audit = chronology(inner_train, inner_valid, "inner")
    audit["inner_valid_fraction_target"] = INNER_VALID_FRACTION
    audit["inner_valid_is_prior_chronological_oof"] = True
    return inner_train, inner_valid, audit


def segment_logistic_probability(train: pd.DataFrame, target: pd.DataFrame, column: str) -> np.ndarray:
    maximum = 12000 if column == "headline_form_item_text" else 30000
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.995,
        max_features=maximum,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform(train[column].astype(str))
    target_x = vectorizer.transform(target[column].astype(str))
    counts = train.groupby("segment_cluster_id").event_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights /= max(float(weights.mean()), 1e-12)
    model = LogisticRegression(
        C=CLASSIFIER_C,
        class_weight="balanced",
        solver="liblinear",
        max_iter=500,
        random_state=SEED,
    )
    model.fit(train_x, train.y.astype(int), sample_weight=weights)
    probability = model.predict_proba(target_x)[:, 1]
    require(np.isfinite(probability).all(), f"{column} model returned non-finite probability")
    return probability


def retrieval_probability(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform(train.material_ex99_text.astype(str))
    target_x = vectorizer.transform(target.material_ex99_text.astype(str))
    similarity = (target_x @ train_x.T).toarray().astype(np.float32, copy=False)
    train_groups: dict[str, list[int]] = {}
    for position, group in enumerate(train.event_group_id.astype(str)):
        train_groups.setdefault(group, []).append(position)
    excluded_pairs = 0
    probabilities = np.empty(len(target), dtype=float)
    top_similarities: list[float] = []
    prior = float(train.y.mean())
    labels = train.y.to_numpy(float)
    k = min(RETRIEVAL_NEIGHBORS, len(train))
    for row_number, group in enumerate(target.event_group_id.astype(str)):
        row = similarity[row_number]
        excluded = train_groups.get(group, [])
        if excluded:
            row[np.asarray(excluded, dtype=int)] = -np.inf
            excluded_pairs += len(excluded)
        selected = np.argpartition(row, -k)[-k:]
        selected_similarity = row[selected]
        valid = np.isfinite(selected_similarity) & (selected_similarity > 0.0)
        if not valid.any():
            probabilities[row_number] = prior
            top_similarities.append(0.0)
            continue
        selected = selected[valid]
        selected_similarity = selected_similarity[valid].astype(float)
        weights = np.exp(
            (selected_similarity - float(selected_similarity.max())) / RETRIEVAL_TEMPERATURE
        )
        probabilities[row_number] = (
            float(np.dot(weights, labels[selected])) + RETRIEVAL_PRIOR_STRENGTH * prior
        ) / (float(weights.sum()) + RETRIEVAL_PRIOR_STRENGTH)
        top_similarities.append(float(selected_similarity.max()))
    require(np.isfinite(probabilities).all(), "retrieval returned non-finite probability")
    return probabilities, {
        "train_n": len(train),
        "query_n": len(target),
        "neighbors": k,
        "temperature": RETRIEVAL_TEMPERATURE,
        "prior_strength": RETRIEVAL_PRIOR_STRENGTH,
        "train_prior": prior,
        "event_group_pairs_excluded": excluded_pairs,
        "mean_top_similarity": float(np.mean(top_similarities)),
        "past_only_by_split_contract": True,
        "query_labels_used": False,
    }


def expert_probabilities(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    headline = segment_logistic_probability(train, target, "headline_form_item_text")
    material = segment_logistic_probability(train, target, "material_ex99_text")
    retrieval, retrieval_audit = retrieval_probability(train, target)
    return {
        "headline_form_item": headline,
        "material_ex99": material,
        "retrieval": retrieval,
    }, {"retrieval": retrieval_audit}


def blend(parts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    output = np.zeros(len(next(iter(parts.values()))), dtype=float)
    total = 0.0
    for name, weight in weights.items():
        output += float(weight) * logit(np.clip(parts[name], 1e-5, 1.0 - 1e-5))
        total += float(weight)
    require(abs(total - 1.0) < 1e-12, "architecture weights must sum to one")
    return expit(output)


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    champion_lookup: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = champion_lookup.set_index("event_id").loc[inner_valid.event_id, "prob"].to_numpy(float)
    baseline_metrics = metric(inner_valid, baseline)
    experts, expert_audit = expert_probabilities(inner_train, inner_valid)
    parts = {"v69": baseline, **experts}
    trials: list[dict[str, Any]] = [
        {
            "name": "V69_NOOP",
            "weights": {"v69": 1.0},
            "metrics": baseline_metrics,
            "auc_delta": 0.0,
            "balanced_accuracy_delta": 0.0,
            "eligible": True,
            "score": float(2.0 * baseline_metrics["auc"] + baseline_metrics["balanced_accuracy"]),
        }
    ]
    for architecture in ARCHITECTURES:
        probability = blend(parts, architecture["weights"])
        values = metric(inner_valid, probability)
        auc_delta = float(values["auc"] - baseline_metrics["auc"])
        ba_delta = float(values["balanced_accuracy"] - baseline_metrics["balanced_accuracy"])
        trials.append(
            {
                "name": architecture["name"],
                "weights": architecture["weights"],
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
        "selection_rule": "inner-past OOF max 2*AUC+BA with BA delta >= -0.010; fixed coarse architectures",
        "baseline": baseline_metrics,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
        "no_threshold_or_weight_tuning": True,
    }, expert_audit


def run_nested(
    sec: pd.DataFrame,
    champion: pd.DataFrame,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v74_applied"] = False
    diagnostic["v74_architecture"] = "V69_NOOP"
    champion_sec = champion.loc[champion.source_family.eq("US_SEC")].copy()
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    sec_lookup = sec.set_index("event_id")
    for fold in (2, 3, 4):
        outer_champion = champion_sec.loc[champion_sec.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        outer_valid = sec_lookup.loc[outer_champion.event_id].reset_index()
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = sec.loc[sec.event_time_utc < outer_start - EMBARGO].copy()
        outer_groups = set(outer_valid.event_group_id.astype(str))
        outer_train = outer_train.loc[~outer_train.event_group_id.astype(str).isin(outer_groups)].copy()
        outer_audit = chronology(outer_train, outer_valid, f"outer fold {fold}")
        inner_train, inner_valid, inner_audit = inner_partition(sec, champion_sec, outer_start)
        policy, inner_expert_audit = choose_inner(inner_train, inner_valid, champion_sec)
        selected = policy["selected"]
        baseline_outer = outer_champion.prob.to_numpy(float)
        outer_expert_audit: dict[str, Any] = {"skipped_for_exact_noop": True}
        if selected["name"] == "V69_NOOP":
            outer_probability = baseline_outer.copy()
        else:
            experts, outer_expert_audit = expert_probabilities(outer_train, outer_valid)
            outer_probability = blend({"v69": baseline_outer, **experts}, selected["weights"])
        baseline_metrics = metric(outer_valid, baseline_outer)
        candidate_metrics = metric(outer_valid, outer_probability)
        probability_map = dict(zip(outer_valid.event_id, outer_probability))
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v74_applied"] = selected["name"] != "V69_NOOP"
        diagnostic.loc[positions, "v74_architecture"] = selected["name"]
        evidence = outer_champion[
            ["event_id", "event_group_id", "event_time_utc", "fold", "y", "fwd_ret_30m", "high_conf"]
        ].copy()
        evidence["baseline_prob"] = baseline_outer
        evidence["candidate_prob"] = outer_probability
        evidence["baseline_direction"] = baseline_outer >= 0.5
        evidence["candidate_direction"] = outer_probability >= 0.5
        evidence["v74_architecture"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append(
            {
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
                "retrieval_uses_only_outer_train_labels": True,
            }
        )
        print(
            f"[V74] fold={fold} architecture={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
        if smoke:
            break
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "outer evidence repeats an event")
    return diagnostic, evidence, audits


def paired_date_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    work["date"] = pd.to_datetime(work.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby(["fold", "date"], sort=True)]
    require(len(blocks) >= 10, "too few date blocks")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        baseline = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)))
        ba_delta.append(
            float(
                balanced_accuracy_score(y, candidate >= 0.5)
                - balanced_accuracy_score(y, baseline >= 0.5)
            )
        )
    require(len(auc_delta) >= int(0.95 * draws), "bootstrap lost too many samples")
    return {
        "method": "paired outer-fold/date-block bootstrap over locked nested architectures",
        "seed": SEED,
        "requested_draws": draws,
        "effective_draws": len(auc_delta),
        "date_blocks": len(blocks),
        "auc_delta_probability_gt_zero": float(np.mean(np.asarray(auc_delta) > 0.0)),
        "auc_delta_quantiles": {
            str(q): float(np.quantile(auc_delta, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
        "balanced_accuracy_delta_probability_gt_zero": float(np.mean(np.asarray(ba_delta) > 0.0)),
        "balanced_accuracy_delta_quantiles": {
            str(q): float(np.quantile(ba_delta, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    segment_audit: dict[str, Any],
    draws: int,
) -> dict[str, Any]:
    baseline_report = json.loads(V69_ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    candidate_summary = v44.summarize(diagnostic)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(baseline_summary["research_gate"] == canonical_baseline, "V69 canonical gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V74 canonical gate mismatch")
    outer_baseline = metric(evidence, evidence.baseline_prob.to_numpy(float))
    outer_candidate = metric(evidence, evidence.candidate_prob.to_numpy(float))
    bootstrap = paired_date_bootstrap(evidence, draws)
    fold_auc_deltas = np.asarray([audit["outer_auc_delta"] for audit in audits], dtype=float)
    base_source = baseline_summary["metrics"]["by_source_family"]["US_SEC"]
    candidate_source = candidate_summary["metrics"]["by_source_family"]["US_SEC"]
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    non_sec = champion.source_family.ne("US_SEC").to_numpy()
    confidence_exact = np.array_equal(
        champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)
    ) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool))
    material_checks = {
        "outer_evidence_auc_delta_gt_0_003": outer_candidate["auc"] - outer_baseline["auc"] > 0.003,
        "outer_evidence_balanced_accuracy_delta_gt_0": (
            outer_candidate["balanced_accuracy"] - outer_baseline["balanced_accuracy"] > 0.0
        ),
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc_deltas > 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta_probability_gt_zero"] >= 0.75,
        "full_us_sec_auc_delta_gt_0_002": candidate_source["auc"] - base_source["auc"] > 0.002,
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
        "non_us_sec_probabilities_exact": np.array_equal(
            champion.loc[non_sec, "prob"].to_numpy(float), diagnostic.loc[non_sec, "prob"].to_numpy(float)
        ),
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"]
            for audit in audits
        ),
        "retrieval_is_past_only_all_folds": all(
            audit["retrieval_uses_only_outer_train_labels"] for audit in audits
        ),
        "focused_segments_only_no_full_document_model_input": (
            not segment_audit["full_document_is_model_input"]
            and segment_audit["observed_max_model_input_chars"] <= 7200
        ),
    }
    material_pass = bool(all(material_checks.values()))
    selected = diagnostic if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(selected_summary["research_gate"] == canonical_selected, "selected canonical gate mismatch")
    exact_fallback = (
        np.array_equal(selected.prob.to_numpy(float), champion.prob.to_numpy(float))
        and np.array_equal(selected.high_conf.to_numpy(bool), champion.high_conf.to_numpy(bool))
        and np.array_equal(
            selected.confidence_signal.to_numpy(float), champion.confidence_signal.to_numpy(float)
        )
    )
    require(material_pass or exact_fallback, "V74 fallback is not exact V69")
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
        "positive_outer_fold_auc_fraction": float(np.mean(fold_auc_deltas > 0.0)),
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
            "policy": "exact V69 prob/confidence_signal/high_conf" if not material_pass else None,
            "exact_arrays_verified": bool(material_pass or exact_fallback),
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
    segment_audit: dict[str, Any],
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
        "Immutable V36 DEV and committed V69 only; focused SEC headline/form/item/material/EX-99 segments; "
        "past-only semantic-response retrieval; outer folds 2-4; inner-only fixed coarse architecture selection; "
        "strict 35-minute embargo; V69 confidence/high_conf frozen; canonical controller gate recomputed."
    )
    report = {
        "version": "V74",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "material_change": "focused SEC semantic segments and past-only response retrieval for direction",
        "disjoint_from": {
            "V71": "no issuer/form/event/taxonomy target-response prior features",
            "V57": "no transformer and no full-document model input",
        },
        "architectures": list(ARCHITECTURES),
        "fold_audits": audits,
        "segment_audit": segment_audit,
        "input_audit": input_audit,
        "evaluation": compact,
        "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V74",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "opened_dev_only": True,
        "reserved_outcomes_loaded": False,
        "selected": selected_contract,
        "candidate": evaluation["candidate_summary"],
        "baseline": evaluation["baseline_summary"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    gate_audit = {
        "version": "V74",
        "status": "MATCH",
        "reported": evaluation["selected_summary"]["research_gate"],
        "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        "material_gate": selected_contract["material_gate"],
    }
    model_comparison = {
        "version": "V74",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "models": {
            "V69_CHAMPION": evaluation["baseline_summary"],
            "V74_DIAGNOSTIC": evaluation["candidate_summary"],
            "V74_FAIL_CLOSED_SELECTED": selected_contract,
        },
        "evaluation": compact,
        "input_audit": input_audit,
        "segment_audit": segment_audit,
        "fold_audits": audits,
        "seal_authorized": False,
    }
    source_transfer = {
        "version": "V74",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "selected_by_source_family": selected_contract["metrics"]["by_source_family"],
        "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
        "selected_by_market": selected_contract["metrics"]["by_market"],
        "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
        "us_sec_outer": {
            "auc_delta": evaluation["outer_auc_delta"],
            "balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
            "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta_probability_gt_zero"],
        },
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    fold_rows = []
    for audit in audits:
        selected = audit["policy"]["selected"]
        fold_rows.append(
            {
                "fold": audit["fold"],
                "outer_train_n": audit["outer_chronology"]["train_n"],
                "outer_valid_n": audit["outer_chronology"]["valid_n"],
                "inner_train_n": audit["inner_chronology"]["train_n"],
                "inner_valid_n": audit["inner_chronology"]["valid_n"],
                "selected_architecture": selected["name"],
                "inner_auc_delta": selected["auc_delta"],
                "inner_ba_delta": selected["balanced_accuracy_delta"],
                "outer_auc_delta": audit["outer_auc_delta"],
                "outer_ba_delta": audit["outer_balanced_accuracy_delta"],
                "strict_35m_embargo": True,
                "outer_labels_used_for_selection": False,
            }
        )
    json_reports = {
        "MODEL_COMPARISON.json": model_comparison,
        "V74_US_SEC_MATERIAL_SEGMENT_RETRIEVAL_REPORT.json": report,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": source_transfer,
        "CANONICAL_RESEARCH_GATE_AUDIT.json": gate_audit,
        "SEC_SEGMENT_AUDIT.json": segment_audit,
        "BOOTSTRAP_DIRECTION_DELTA_REPORT.json": evaluation["bootstrap"],
    }
    selected_columns = list(champion.columns)
    table_reports = {
        "V74_US_SEC_FOLD_AUDIT.csv": pd.DataFrame(fold_rows),
        "V74_US_SEC_DIRECTION_EVIDENCE_OOF.csv.gz": evidence,
        "V74_DIAGNOSTIC_CHALLENGER_OOF.csv.gz": evaluation["diagnostic_frame"],
        "V74_FAIL_CLOSED_SELECTED_OOF.csv.gz": evaluation["selected_frame"][selected_columns],
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
        "classifier_c": CLASSIFIER_C,
        "retrieval_neighbors": RETRIEVAL_NEIGHBORS,
        "retrieval_temperature": RETRIEVAL_TEMPERATURE,
        "retrieval_prior_strength": RETRIEVAL_PRIOR_STRENGTH,
        "architectures": ARCHITECTURES,
        "embargo_seconds": int(EMBARGO.total_seconds()),
        "validation": "nested chronological OOF",
    }
    experiment_id = hashlib.sha256(
        (
            "|".join(input_audit["input_sha256"].values())
            + "|"
            + json.dumps(material_spec, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    run_status = {
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
    atomic_json(run_status, OUT / "RUN_STATUS.json")
    files: dict[str, Any] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}:
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "experiment_id": experiment_id,
        "files": files,
    }
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
    sec, champion, input_audit = verify_and_load()
    sec, segment_audit = prepare_sec(sec)
    if args.audit_only:
        print(
            json.dumps(
                clean(
                    {
                        "status": "AUDIT_OK",
                        "hypothesis": HYPOTHESIS,
                        "input_audit": input_audit,
                        "segment_audit": segment_audit,
                        "architectures": list(ARCHITECTURES),
                        "output_written": False,
                    }
                ),
                indent=2,
            )
        )
        return
    diagnostic, evidence, audits = run_nested(sec, champion, smoke=args.smoke_test)
    evaluation = evaluate(
        champion,
        diagnostic,
        evidence,
        audits,
        segment_audit,
        draws=250 if args.smoke_test else BOOTSTRAP_DRAWS,
    )
    summary = {
        "status": evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "material_pass": evaluation["material_pass"],
        "selected_architectures": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
        "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta_probability_gt_zero"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    if args.smoke_test:
        print(json.dumps(clean({**summary, "status": "SMOKE_OK", "output_written": False}), indent=2))
        return
    json_reports, table_reports = build_reports(
        champion, evidence, audits, evaluation, input_audit, segment_audit
    )
    if args.dry_run:
        print(json.dumps(clean({**summary, "status": "DRY_RUN_OK", "output_written": False}), indent=2))
        return
    write_result = write_outputs(json_reports, table_reports, evaluation, input_audit)
    print(json.dumps(clean({**summary, "output_dir": str(OUT), "write": write_result}), indent=2))


if __name__ == "__main__":
    main()
