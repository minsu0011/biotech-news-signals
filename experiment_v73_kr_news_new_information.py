"""V73 material KR_NEWS challenger using genuinely new Korean information.

Authority is deliberately narrow: the immutable V36 DEV file and the atomic
``output_V69`` commit.  The runner never names or opens DEV-extension, role,
seal, or final-reserve assets.  V69 is the exact fail-closed champion.

The challenger separates headline and body representations, combines Korean
word and character TF-IDF, adds deterministic event/biotech-sector taxonomy,
past-only issuer/source/taxonomy response priors, and uses near-duplicate
clusters for both leakage exclusion and training weights.  A coarse model and
blend architecture is selected only on embargoed inner-past OOF evidence.
Outer labels are evaluation-only.  The controller's canonical 14 checks and
confidence-correctness AUC are recomputed from raw selected-frame evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, logit
from sklearn.feature_extraction import DictVectorizer
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
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "staging" / "V73_KR_NEWS_NEW_INFORMATION_V1"
)
CACHE_DIR = ROOT / "research" / "V73_KR_NEWS_NEW_INFORMATION_FULL_CACHE"

VERSION = 73
HYPOTHESIS = "KR_NEWS_NEW_INFORMATION_HEAD_BODY_CAUSAL_CONTEXT_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_CACHE_MANIFEST_SHA256 = "3c975094e2f7e472c8a50e62e849d9609646b39452fc6285e3eb1119c0df1925"
EXPECTED_CACHE_EXPERIMENT_ID = "e788771632e19160f27e29da0484848e3891cded1011e6ba4573d16f1009d7b3"
EXPECTED_V36_ROWS = 9462
EXPECTED_V69_ROWS = 7568
EXPECTED_KR_NEWS_OOF_ROWS = 1139
EXPECTED_KR_NEWS_DEV_ROWS = 1425
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
PRIOR_STRENGTH = 14.0
SEED = 7301
BOOTSTRAP_DRAWS = 2000
COST = 0.002
BLEND_WEIGHTS = (0.35, 0.60, 0.85)

CONTEXT_COLUMNS = (
    "pre_ret_2", "pre_ret_5", "pre_ret_15", "pre_ret_30", "pre_ret_60",
    "pre_vol_5", "pre_vol_30", "volume_ratio_2_30", "volume_ratio_5_30",
    "range_10m", "range_30m", "close_position_30", "intraday_ret_open",
    "return_autocorr_30", "trend_slope_30", "trend_slope_60",
    "minutes_from_open", "minutes_to_close", "benchmark_ret_5",
    "benchmark_ret_15", "benchmark_ret_30", "event_positive_kw",
    "event_negative_kw", "event_financing_kw", "event_trial_kw",
    "event_regulatory_kw", "event_ma_kw", "event_earnings_kw",
)

MODEL_SPECS = (
    {"name": "HEADLINE_TAXONOMY", "body": False, "context": False},
    {"name": "SEPARATE_HEADLINE_BODY", "body": True, "context": False},
    {"name": "HEAD_BODY_CAUSAL_CONTEXT", "body": True, "context": True},
)

EVENT_PATTERNS = {
    "REGULATORY": r"허가|승인|식약처|fda|품목허가|신속심사|우선심사|보완요구|반려",
    "CLINICAL": r"임상|시험|환자|투약|등록|1상|2상|3상|유효성|안전성|중간결과|탑라인",
    "FINANCING": r"유상증자|무상증자|전환사채|신주인수권|교환사채|자금조달|공모|사모|희석",
    "PARTNERSHIP": r"기술수출|라이선스|계약|협약|공동개발|파트너|마일스톤|로열티",
    "EARNINGS": r"실적|매출|영업이익|순이익|적자|흑자|전망|가이던스|분기|결산",
    "MA_GOVERNANCE": r"인수|합병|매각|최대주주|대표이사|경영권|이사회|주주총회",
    "PRODUCT_OPERATIONS": r"출시|공급|수주|생산|공장|시설|리콜|특허|소송",
}

SECTOR_PATTERNS = {
    "ONCOLOGY": r"암|항암|종양|면역항암|car[- ]?t",
    "NEURO": r"치매|알츠하이머|파킨슨|뇌|신경|우울",
    "METABOLIC": r"비만|당뇨|대사|지방간|glp[- ]?1",
    "IMMUNE_RARE": r"자가면역|희귀질환|염증|아토피|류마티스",
    "INFECTIOUS_VACCINE": r"백신|감염|바이러스|세균|진단키트",
    "CELL_GENE": r"세포치료|유전자치료|줄기세포|rna|유전체|크리스퍼",
    "DEVICE_DIAGNOSTIC": r"의료기기|진단|영상|검사|바이오마커|ai 의료",
}

WORD_RE = re.compile(r"[가-힣]{1,}|[A-Za-z]{2,}|\d+(?:\.\d+)?(?:상|기|%|억|조)?")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.map(lambda item: str(item).strip().lower() in {"1", "true", "yes"}).astype(bool)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, int)
    return float(roc_auc_score(target, np.asarray(score, float))) if np.unique(target).size == 2 else None


def normalize_korean_text(value: Any, number_token: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = URL_RE.sub(" URL ", text)
    if number_token:
        text = NUMBER_RE.sub(" NUM ", text)
    text = " ".join(WORD_RE.findall(text))
    return SPACE_RE.sub(" ", text).strip()


def korean_tokenize(value: str) -> list[str]:
    return WORD_RE.findall(normalize_korean_text(value, number_token=False))


def taxonomy_label(text: str, patterns: dict[str, str]) -> str:
    hits = [name for name, pattern in patterns.items() if re.search(pattern, text, flags=re.IGNORECASE)]
    return "+".join(hits) if hits else "OTHER"


def near_duplicate_id(row: pd.Series) -> str:
    headline = normalize_korean_text(row.get("headline", ""), number_token=True)
    # A prefix absorbs syndicated copies with a short publisher suffix while
    # retaining issuer/time separation through the ticker and 30-minute bin.
    tokens = headline.split()[:28]
    timestamp = pd.Timestamp(row.event_time_utc).floor("30min").isoformat()
    payload = f"{row.ticker}|{timestamp}|{' '.join(tokens)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_authority() -> dict[str, Any]:
    commit_path = V69_DIR / "COMMIT.json"
    manifest_path = V69_DIR / "ARTIFACT_MANIFEST.json"
    require(sha256(DEV_PATH) == EXPECTED_V36_SHA256, "immutable V36 DEV hash changed")
    require(commit_path.is_file() and manifest_path.is_file(), "V69 authority is incomplete")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment changed")
    require(commit.get("manifest_sha256") == sha256(manifest_path), "V69 manifest binding changed")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest experiment changed")
    for name, item in manifest.get("files", {}).items():
        path = V69_DIR / name
        require(path.is_file(), f"V69 manifested artifact missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V69 artifact size changed: {name}")
        require(sha256(path) == item["sha256"], f"V69 artifact hash changed: {name}")
    require(V69_PATH.name in manifest.get("files", {}), "V69 selected OOF is not manifested")
    require(commit.get("data_sha") == EXPECTED_V36_SHA256, "V69 is not bound to immutable V36 DEV")
    return {
        "authority": "immutable V36 DEV + atomic output_V69",
        "authorized_inputs": [
            str(DEV_PATH.relative_to(ROOT)),
            str(commit_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)),
            str(V69_PATH.relative_to(ROOT)),
        ],
        "v36_dev_sha256": EXPECTED_V36_SHA256,
        "v69_commit_sha256": sha256(commit_path),
        "v69_manifest_sha256": sha256(manifest_path),
        "v69_selected_oof_sha256": sha256(V69_PATH),
        "experiment_id": EXPECTED_V69_EXPERIMENT_ID,
        "manifest_files_verified": len(manifest.get("files", {})),
        "opened_dev_only": True,
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "research_seal_loaded": False,
        "final_reserve_loaded": False,
    }


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    champion = pd.read_csv(
        V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_V36_ROWS, "immutable V36 row count changed")
    require(len(champion) == EXPECTED_V69_ROWS, "V69 OOF row count changed")
    require(dev.event_id.is_unique and champion.event_id.is_unique, "event_id uniqueness failed")
    require(set(champion.event_id).issubset(set(dev.event_id)), "V69 OOF is not a V36 DEV subset")
    aligned = dev.set_index("event_id").loc[champion.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion.y.to_numpy(int)), "V36/V69 labels differ")
    require(np.allclose(aligned.fwd_ret_30m, champion.fwd_ret_30m, rtol=0.0, atol=1e-15), "V36/V69 returns differ")
    require(aligned.source_family.astype(str).equals(champion.source_family.astype(str)), "V36/V69 source differs")
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    champion["high_conf"] = bool_series(champion.high_conf)
    kr = dev.loc[dev.source_family.eq("KR_NEWS")].copy()
    kr["event_time_utc"] = pd.to_datetime(kr.event_time_utc, utc=True)
    kr["headline"] = kr.headline.fillna("").astype(str)
    kr["body"] = kr.body.fillna("").astype(str)
    kr["source"] = kr.source.fillna("UNKNOWN").astype(str)
    kr["headline_norm"] = kr.headline.map(normalize_korean_text)
    kr["body_norm"] = kr.body.str.slice(0, 5000).map(normalize_korean_text)
    combined = kr.headline_norm + " " + kr.body_norm.str.slice(0, 1600)
    kr["event_taxonomy"] = combined.map(lambda text: taxonomy_label(text, EVENT_PATTERNS))
    kr["sector_taxonomy"] = combined.map(lambda text: taxonomy_label(text, SECTOR_PATTERNS))
    kr["source_key"] = kr.source.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    kr["near_duplicate_id"] = kr.apply(near_duplicate_id, axis=1)
    kr = kr.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(len(kr) == EXPECTED_KR_NEWS_DEV_ROWS, f"KR_NEWS DEV count changed: {len(kr)}")
    require(kr.y.isin([0, 1]).all(), "invalid KR_NEWS direction label")
    require(int(champion.source_family.eq("KR_NEWS").sum()) == EXPECTED_KR_NEWS_OOF_ROWS, "KR_NEWS OOF count changed")
    return kr, champion


def group_keys(row: pd.Series) -> dict[str, str]:
    ticker = str(row.ticker)
    source = str(row.source_key)
    event = str(row.event_taxonomy)
    sector = str(row.sector_taxonomy)
    return {
        "issuer": ticker,
        "source": source,
        "event": event,
        "sector": sector,
        "source_event": f"{source}|{event}",
        "issuer_sector": f"{ticker}|{sector}",
    }


def smoothed(total: float, count: int, global_prior: float) -> float:
    return float((total + PRIOR_STRENGTH * global_prior) / (count + PRIOR_STRENGTH))


def prior_row(row: pd.Series, stats: dict[str, dict[str, list[float]]], global_prior: float) -> dict[str, float]:
    output: dict[str, float] = {"prior::global": global_prior}
    for family, key in group_keys(row).items():
        total, count = stats[family].get(key, [0.0, 0.0])
        value = smoothed(total, int(count), global_prior)
        output[f"prior::{family}"] = value
        output[f"prior::{family}::delta"] = value - global_prior
        output[f"prior::{family}::log_count"] = float(np.log1p(count))
        if family == "source":
            output["source_quality::reliability"] = float(count / (count + PRIOR_STRENGTH))
            output["source_quality::response_abs_delta"] = abs(value - global_prior)
    return output


def update_stats(stats: dict[str, dict[str, list[float]]], row: pd.Series) -> None:
    for family, key in group_keys(row).items():
        cell = stats[family][key]
        cell[0] += float(row.y)
        cell[1] += 1.0


def causal_prior_features(train: pd.DataFrame, target: pd.DataFrame) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable")
    stats: dict[str, dict[str, list[float]]] = {
        family: defaultdict(lambda: [0.0, 0.0])
        for family in ("issuer", "source", "event", "sector", "source_event", "issuer_sector")
    }
    features_by_index: dict[Any, dict[str, float]] = {}
    total = 0.0
    count = 0
    for index, row in ordered.iterrows():
        global_prior = float((total + PRIOR_STRENGTH * 0.5) / (count + PRIOR_STRENGTH))
        features_by_index[index] = prior_row(row, stats, global_prior)
        update_stats(stats, row)
        total += float(row.y)
        count += 1
    global_prior = float((total + PRIOR_STRENGTH * 0.5) / (count + PRIOR_STRENGTH))
    return [features_by_index[index] for index in train.index], [prior_row(row, stats, global_prior) for _, row in target.iterrows()]


def structured_features(frame: pd.DataFrame, priors: list[dict[str, float]], context: bool) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        values = dict(priors[position])
        values[f"cat::source={row.source_key}"] = 1.0
        values[f"cat::event={row.event_taxonomy}"] = 1.0
        values[f"cat::sector={row.sector_taxonomy}"] = 1.0
        values[f"cat::event_sector={row.event_taxonomy}|{row.sector_taxonomy}"] = 1.0
        values["text::headline_log_length"] = float(np.log1p(len(row.headline_norm)))
        values["text::body_log_length"] = float(np.log1p(len(row.body_norm)))
        values["text::body_present"] = float(bool(row.body_norm))
        if context:
            for column in CONTEXT_COLUMNS:
                value = finite(row.get(column))
                if value is not None:
                    values[f"context::{column}"] = float(np.clip(value, -20.0, 20.0))
        output.append(values)
    return output


class KoreanInformationModel:
    def __init__(self, spec: dict[str, Any], seed: int):
        self.spec = dict(spec)
        self.seed = int(seed)
        self.head_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_df=0.997,
            max_features=42000, sublinear_tf=True, dtype=np.float32,
        )
        self.head_word = TfidfVectorizer(
            tokenizer=korean_tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), min_df=2, max_df=0.997, max_features=30000,
            sublinear_tf=True, dtype=np.float32,
        )
        self.body_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_df=0.997,
            max_features=35000, sublinear_tf=True, dtype=np.float32,
        )
        self.body_word = TfidfVectorizer(
            tokenizer=korean_tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), min_df=3, max_df=0.997, max_features=30000,
            sublinear_tf=True, dtype=np.float32,
        )
        self.dictionary = DictVectorizer(sparse=True, dtype=np.float64)
        self.model = LogisticRegression(
            C=0.35, solver="liblinear", class_weight="balanced", max_iter=700,
            random_state=self.seed,
        )

    def fit_predict(self, train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        train_prior, target_prior = causal_prior_features(train, target)
        train_parts = [
            self.head_char.fit_transform(train.headline_norm),
            self.head_word.fit_transform(train.headline_norm),
        ]
        target_parts = [
            self.head_char.transform(target.headline_norm),
            self.head_word.transform(target.headline_norm),
        ]
        if self.spec["body"]:
            train_parts.extend([
                self.body_char.fit_transform(train.body_norm),
                self.body_word.fit_transform(train.body_norm),
            ])
            target_parts.extend([
                self.body_char.transform(target.body_norm),
                self.body_word.transform(target.body_norm),
            ])
        train_structured = structured_features(train, train_prior, bool(self.spec["context"]))
        target_structured = structured_features(target, target_prior, bool(self.spec["context"]))
        train_parts.append(self.dictionary.fit_transform(train_structured))
        target_parts.append(self.dictionary.transform(target_structured))
        train_matrix = hstack(train_parts, format="csr", dtype=np.float64)
        target_matrix = hstack(target_parts, format="csr", dtype=np.float64)
        counts = train.groupby("near_duplicate_id").event_id.transform("size").to_numpy(float)
        sample_weight = 1.0 / np.maximum(counts, 1.0)
        self.model.fit(train_matrix, train.y.to_numpy(int), sample_weight=sample_weight)
        probability = self.model.predict_proba(target_matrix)[:, 1]
        require(np.isfinite(probability).all(), "Korean information model produced non-finite probability")
        return probability, {
            "spec": self.spec,
            "train_rows": len(train),
            "target_rows": len(target),
            "feature_n": train_matrix.shape[1],
            "headline_char_features": train_parts[0].shape[1],
            "headline_word_features": train_parts[1].shape[1],
            "body_separate": bool(self.spec["body"]),
            "past_only_causal_priors": True,
            "near_duplicate_inverse_weighting": True,
        }


def probability_blend(baseline: np.ndarray, challenger: np.ndarray, weight: float) -> np.ndarray:
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    new_logit = logit(np.clip(np.asarray(challenger, float), 1e-5, 1.0 - 1e-5))
    return expit((1.0 - weight) * base_logit + weight * new_logit)


def metric(frame: pd.DataFrame, probability: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    y = frame.y.to_numpy(int)
    correct = (prediction == y).astype(int)
    return {
        "n": len(frame),
        "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(prediction == y)),
        "pred_up": float(prediction.mean()),
        "confidence_correctness_auc": safe_auc(correct, np.asarray(confidence, float)),
    }


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) and len(valid), f"{label}: empty split")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: strict 35-minute embargo failed")
    event_overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    duplicate_overlap = set(train.near_duplicate_id.astype(str)) & set(valid.near_duplicate_id.astype(str))
    require(not event_overlap, f"{label}: event-group leakage")
    require(not duplicate_overlap, f"{label}: near-duplicate leakage")
    return {
        "train_n": len(train),
        "valid_n": len(valid),
        "train_end": train_end,
        "valid_start": valid_start,
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "event_group_overlap_n": 0,
        "near_duplicate_overlap_n": 0,
        "strict_35m_embargo": True,
    }


def leakage_filtered_train(frame: pd.DataFrame, boundary: pd.Timestamp, valid: pd.DataFrame) -> pd.DataFrame:
    train = frame.loc[frame.event_time_utc < boundary - EMBARGO].copy()
    train = train.loc[~train.event_group_id.astype(str).isin(set(valid.event_group_id.astype(str)))].copy()
    train = train.loc[~train.near_duplicate_id.astype(str).isin(set(valid.near_duplicate_id.astype(str)))].copy()
    return train


def inner_partition(kr: pd.DataFrame, champion_kr: pd.DataFrame, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible = champion_kr.loc[champion_kr.event_time_utc < outer_start - EMBARGO].copy()
    eligible = eligible.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(eligible) >= 220, "insufficient prior KR_NEWS OOF evidence")
    boundary_position = max(120, int(math.floor(len(eligible) * (1.0 - INNER_VALID_FRACTION))))
    inner_valid_ids = set(eligible.iloc[boundary_position:].event_id)
    inner_valid = kr.loc[kr.event_id.isin(inner_valid_ids)].copy()
    inner_valid = inner_valid.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(inner_valid) >= 40, "inner KR_NEWS validation is too small")
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = leakage_filtered_train(kr, inner_start, inner_valid)
    require(len(inner_train) >= 100 and inner_train.y.nunique() == 2, "inner KR_NEWS train is insufficient")
    audit = chronology(inner_train, inner_valid, "V73 inner")
    audit["validation_source"] = "prior committed V69 OOF event IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, audit


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, champion_kr: pd.DataFrame, fold: int) -> dict[str, Any]:
    lookup = champion_kr.set_index("event_id")
    baseline = lookup.loc[inner_valid.event_id, "prob"].to_numpy(float)
    confidence = lookup.loc[inner_valid.event_id, "confidence_signal"].to_numpy(float)
    baseline_metric = metric(inner_valid, baseline, confidence)
    trials: list[dict[str, Any]] = [{
        "name": "V69_NOOP", "spec": None, "challenger_weight": 0.0,
        "metrics": baseline_metric, "auc_delta": 0.0,
        "balanced_accuracy_delta": 0.0, "confidence_correctness_auc_delta": 0.0,
        "eligible": True,
        "score": float(1.5 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 0.35 * (baseline_metric["confidence_correctness_auc"] or 0.5)),
        "model_audit": None,
    }]
    for offset, spec in enumerate(MODEL_SPECS):
        challenger, model_audit = KoreanInformationModel(spec, SEED + fold * 10 + offset).fit_predict(inner_train, inner_valid)
        for weight in BLEND_WEIGHTS:
            probability = probability_blend(baseline, challenger, float(weight))
            values = metric(inner_valid, probability, confidence)
            auc_delta = values["auc"] - baseline_metric["auc"]
            ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
            cc_delta = (values["confidence_correctness_auc"] or 0.0) - (baseline_metric["confidence_correctness_auc"] or 0.0)
            trials.append({
                "name": f"{spec['name']}_W{weight:.2f}",
                "spec": spec,
                "challenger_weight": float(weight),
                "metrics": values,
                "auc_delta": auc_delta,
                "balanced_accuracy_delta": ba_delta,
                "confidence_correctness_auc_delta": cc_delta,
                "eligible": bool(ba_delta >= -0.010 and cc_delta >= -0.020),
                "score": float(1.5 * values["auc"] + values["balanced_accuracy"] + 0.35 * (values["confidence_correctness_auc"] or 0.5)),
                "model_audit": model_audit,
            })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["confidence_correctness_auc_delta"], -trial["challenger_weight"]),
    )
    return {
        "selection_rule": "maximize 1.5*AUC + BA + 0.35*confidence-correctness-AUC on inner-past OOF; BA delta >= -0.010 and confidence AUC delta >= -0.020; coarse weights only",
        "baseline": baseline_metric,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
    }


def run_nested(kr: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v73_applied"] = False
    diagnostic["v73_model"] = "V69_NOOP"
    diagnostic["v73_challenger_weight"] = 0.0
    champion_kr = champion.loc[champion.source_family.eq("KR_NEWS")].copy()
    champion_kr["event_time_utc"] = pd.to_datetime(champion_kr.event_time_utc, utc=True)
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold in (2, 3, 4):
        outer_champion = champion_kr.loc[champion_kr.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        outer_valid = kr.set_index("event_id").loc[outer_champion.event_id].reset_index()
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = leakage_filtered_train(kr, outer_start, outer_valid)
        require(len(outer_train) >= 200 and outer_train.y.nunique() == 2, f"outer fold {fold} train insufficient")
        outer_audit = chronology(outer_train, outer_valid, f"V73 outer fold {fold}")
        inner_train, inner_valid, inner_audit = inner_partition(kr, champion_kr, outer_start)
        policy = choose_inner(inner_train, inner_valid, champion_kr, fold)
        selected = policy["selected"]
        baseline_outer = outer_champion.prob.to_numpy(float)
        confidence_outer = outer_champion.confidence_signal.to_numpy(float)
        outer_model_audit = None
        if selected["spec"] is None:
            outer_probability = baseline_outer.copy()
        else:
            challenger, outer_model_audit = KoreanInformationModel(selected["spec"], SEED + 1000 + fold).fit_predict(outer_train, outer_valid)
            outer_probability = probability_blend(baseline_outer, challenger, float(selected["challenger_weight"]))
        baseline_metric = metric(outer_valid, baseline_outer, confidence_outer)
        candidate_metric = metric(outer_valid, outer_probability, confidence_outer)
        ids = set(outer_valid.event_id)
        positions = diagnostic.index[diagnostic.event_id.isin(ids)]
        probability_map = dict(zip(outer_valid.event_id, outer_probability))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v73_applied"] = selected["spec"] is not None
        diagnostic.loc[positions, "v73_model"] = selected["name"]
        diagnostic.loc[positions, "v73_challenger_weight"] = float(selected["challenger_weight"])
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "ticker", "source_family",
            "fold", "y", "fwd_ret_30m", "high_conf", "confidence_signal",
        ]].copy()
        evidence["baseline_prob"] = baseline_outer
        evidence["candidate_prob"] = outer_probability
        evidence["baseline_correct"] = ((baseline_outer >= 0.5) == evidence.y.to_numpy(int)).astype(int)
        evidence["candidate_correct"] = ((outer_probability >= 0.5) == evidence.y.to_numpy(int)).astype(int)
        evidence["v73_model"] = selected["name"]
        evidence["v73_challenger_weight"] = float(selected["challenger_weight"])
        evidence_parts.append(evidence)
        audits.append({
            "fold": fold,
            "outer_chronology": outer_audit,
            "inner_chronology": inner_audit,
            "policy": policy,
            "outer_model_audit": outer_model_audit,
            "outer_baseline": baseline_metric,
            "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - baseline_metric["auc"],
            "outer_balanced_accuracy_delta": candidate_metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"],
            "outer_confidence_correctness_auc_delta": (candidate_metric["confidence_correctness_auc"] or 0.0) - (baseline_metric["confidence_correctness_auc"] or 0.0),
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V73] fold={fold} model={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f} "
            f"outer_conf_auc_delta={audits[-1]['outer_confidence_correctness_auc_delta']:+.6f}",
            flush=True,
        )
        if smoke:
            break
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V73 outer evidence repeats an event")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    work["date"] = pd.to_datetime(work.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby(["fold", "date"], sort=True)]
    # A one-fold smoke has six calendar blocks in the immutable contract; the
    # full three-fold run has materially more.  Five is therefore a structural
    # smoke floor, not a relaxation of the full material checks.
    require(len(blocks) >= 5, "too few V73 date blocks")
    generator = np.random.default_rng(SEED)
    values = {"auc": [], "ba": [], "confidence_correctness_auc": []}
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        confidence = sample.confidence_signal.to_numpy(float)
        base_correct = ((base >= 0.5) == y).astype(int)
        candidate_correct = ((candidate >= 0.5) == y).astype(int)
        values["auc"].append(float(roc_auc_score(y, candidate) - roc_auc_score(y, base)))
        values["ba"].append(float(balanced_accuracy_score(y, candidate >= 0.5) - balanced_accuracy_score(y, base >= 0.5)))
        base_cc = safe_auc(base_correct, confidence)
        candidate_cc = safe_auc(candidate_correct, confidence)
        if base_cc is not None and candidate_cc is not None:
            values["confidence_correctness_auc"].append(candidate_cc - base_cc)
    require(len(values["auc"]) >= int(0.90 * draws), "V73 bootstrap lost too many samples")
    return {
        "method": "paired outer-fold/date-block bootstrap over locked nested policies",
        "seed": SEED,
        "requested_draws": draws,
        "date_blocks": len(blocks),
        "deltas": {
            name: {
                "effective_draws": len(samples),
                "probability_gt_zero": float(np.mean(np.asarray(samples) > 0.0)),
                "lower95": float(np.quantile(samples, 0.025)),
                "median": float(np.median(samples)),
                "upper95": float(np.quantile(samples, 0.975)),
            }
            for name, samples in values.items()
        },
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    candidate_summary = v44.summarize(diagnostic)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == 14 and canonical_candidate["total"] == 14, "canonical controller gate is not 14 checks")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller canonical gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V73/controller canonical gate mismatch")
    outer_base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal)
    outer_candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal)
    bootstrap = paired_bootstrap(evidence, draws)
    base_source = baseline_summary["metrics"]["by_source_family"]["KR_NEWS"]
    candidate_source = candidate_summary["metrics"]["by_source_family"]["KR_NEWS"]
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_cc = np.asarray([audit["outer_confidence_correctness_auc_delta"] for audit in audits], float)
    non_kr = champion.source_family.ne("KR_NEWS").to_numpy(bool)
    confidence_exact = np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool))
    checks = {
        "outer_auc_delta_gt_0_003": outer_candidate["auc"] - outer_base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": outer_candidate["balanced_accuracy"] - outer_base["balanced_accuracy"] > 0.0,
        "outer_confidence_correctness_auc_delta_gt_0_005": (outer_candidate["confidence_correctness_auc"] or 0.0) - (outer_base["confidence_correctness_auc"] or 0.0) > 0.005,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_confidence_auc_fraction_ge_2_of_3": float(np.mean(fold_cc >= 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["deltas"]["auc"]["probability_gt_zero"] >= 0.75,
        "full_kr_news_auc_delta_gt_0_002": candidate_source["auc"] - base_source["auc"] > 0.002,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "non_kr_news_probabilities_exact": np.array_equal(champion.loc[non_kr, "prob"].to_numpy(float), diagnostic.loc[non_kr, "prob"].to_numpy(float)),
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "new_information_blocks_present": all(audit["policy"]["selected"]["spec"] is None or audit["policy"]["selected"]["model_audit"]["past_only_causal_priors"] for audit in audits),
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    selected_canonical = controller.canonical_research_gate(selected_summary)
    require(selected_canonical["total"] == 14 and selected_summary["research_gate"] == selected_canonical, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V73 fail-closed frame is not exact V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass,
        "material_checks": checks,
        "outer_baseline": outer_base,
        "outer_candidate": outer_candidate,
        "outer_auc_delta": outer_candidate["auc"] - outer_base["auc"],
        "outer_balanced_accuracy_delta": outer_candidate["balanced_accuracy"] - outer_base["balanced_accuracy"],
        "outer_confidence_correctness_auc_delta": (outer_candidate["confidence_correctness_auc"] or 0.0) - (outer_base["confidence_correctness_auc"] or 0.0),
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": selected_canonical, "reported_equals_controller_recomputed": True},
        "direction_change": {
            "probability_changed_n": int(np.sum(champion.prob.to_numpy(float) != diagnostic.prob.to_numpy(float))),
            "direction_changed_n": int(np.sum((champion.prob.to_numpy(float) >= 0.5) != (diagnostic.prob.to_numpy(float) >= 0.5))),
            "baseline_probability_sha256": array_sha256(champion.prob.to_numpy(float)),
            "candidate_probability_sha256": array_sha256(diagnostic.prob.to_numpy(float)),
            "selected_probability_sha256": array_sha256(selected.prob.to_numpy(float)),
        },
        "fallback": {"activated": not material_pass, "policy": "exact V69 entire frame" if not material_pass else None, "exact_frame_verified": bool(material_pass or selected.equals(champion))},
        "diagnostic_frame": diagnostic,
        "selected_frame": selected,
    }


def write_outputs(authority: dict[str, Any], champion: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V73", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V73_DIAGNOSTIC": evaluation["candidate_summary"], "V73_FAIL_CLOSED_SELECTED": evaluation["selected_summary"]},
            "evaluation": compact, "authority_audit": authority, "nested_fold_audits": audits,
            "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V73", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": evaluation["selected_summary"],
            "candidate": evaluation["candidate_summary"], "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"], "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V73", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "kr_news_outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
                "confidence_correctness_auc_delta": evaluation["outer_confidence_correctness_auc_delta"],
            },
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "KR_NEWS_NEW_INFORMATION_REPORT.json": {
            "version": "V73", "hypothesis": HYPOTHESIS,
            "information_blocks": ["separate headline/body", "Korean word+character TF-IDF", "event taxonomy", "biotech sector taxonomy", "past-only source quality", "past-only issuer/sector response priors", "near-duplicate exclusion+weighting", "pre-event context"],
            "model_specs": MODEL_SPECS, "blend_weights": BLEND_WEIGHTS,
            "confidence_correctness_auc_reported": True, "evaluation": compact,
            "nested_fold_audits": audits, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V73", "status": "MATCH", "canonical_check_count": 14,
            "reported": evaluation["selected_summary"]["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if evaluation["selected_summary"]["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False, "completed_at": now(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    fold_rows = [{
        "fold": audit["fold"], "selected_model": audit["policy"]["selected"]["name"],
        "selected_weight": audit["policy"]["selected"]["challenger_weight"],
        "inner_auc_delta": audit["policy"]["selected"]["auc_delta"],
        "inner_confidence_correctness_auc_delta": audit["policy"]["selected"]["confidence_correctness_auc_delta"],
        "outer_auc_delta": audit["outer_auc_delta"],
        "outer_ba_delta": audit["outer_balanced_accuracy_delta"],
        "outer_confidence_correctness_auc_delta": audit["outer_confidence_correctness_auc_delta"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]
    atomic_csv(pd.DataFrame(fold_rows), out / "V73_KR_NEWS_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V73_KR_NEWS_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V73_DIAGNOSTIC_CHALLENGER_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"][list(champion.columns)], out / "V73_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}
    atomic_json(manifest, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def adopt_verified_full_cache(authority: dict[str, Any], champion: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Adopt the already completed deterministic full run after contract repair.

    The first controller execution completed every V73 fold and wrote a bound
    manifest, but the controller stopped before COMMIT because the runner had
    omitted SOURCE_TRANSFER_REPORT.json.  Re-fitting the same deterministic
    model would add no evidence, so this path revalidates every cached byte,
    proves the fail-closed selected frame equals V69, and projects only the
    runner-owned artifacts into the new controller stage.
    """
    manifest_path = CACHE_DIR / "ARTIFACT_MANIFEST.json"
    require(manifest_path.is_file(), "V73 completed full-run cache is missing")
    require(sha256(manifest_path) == EXPECTED_CACHE_MANIFEST_SHA256, "V73 cache manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("version") == 73, "V73 cache version mismatch")
    require(manifest.get("experiment_id") == EXPECTED_CACHE_EXPERIMENT_ID, "V73 cache experiment mismatch")
    for name, item in manifest.get("files", {}).items():
        path = CACHE_DIR / name
        require(path.is_file(), f"V73 cached artifact missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V73 cached artifact size changed: {name}")
        require(sha256(path) == item["sha256"], f"V73 cached artifact hash changed: {name}")

    plan = json.loads((CACHE_DIR / "VERSION_PLAN.json").read_text(encoding="utf-8"))
    require(plan.get("hypothesis") == HYPOTHESIS, "V73 cached hypothesis mismatch")
    require(plan.get("data_sha") == EXPECTED_V36_SHA256, "V73 cached DEV authority mismatch")
    require(plan.get("seal_outcomes_loaded") is False, "V73 cache opened research seal")
    require(plan.get("final_reserve_outcomes_loaded") is False, "V73 cache opened final reserve")

    model_report = json.loads((CACHE_DIR / "MODEL_COMPARISON.json").read_text(encoding="utf-8"))
    evaluation = model_report["evaluation"]
    require(evaluation.get("material_pass") is False, "V73 cache is not the recorded failed material trial")
    require(evaluation.get("fallback", {}).get("exact_frame_verified") is True, "V73 cache fallback audit failed")
    selected_report = json.loads((CACHE_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    recomputed_gate = controller.canonical_research_gate(selected_report["selected"])
    require(recomputed_gate.get("total") == 14, "V73 cached canonical gate is not 14 checks")
    require(selected_report["selected"].get("research_gate") == recomputed_gate, "V73 cached gate mismatch")

    cached_selected = pd.read_csv(
        CACHE_DIR / "V73_FAIL_CLOSED_SELECTED_OOF.csv.gz",
        compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    cached_selected["high_conf"] = bool_series(cached_selected.high_conf)
    require(cached_selected.equals(champion), "V73 cached selected frame is not exact V69")

    runner_files = (
        "MODEL_COMPARISON.json", "DEV_ROBUSTNESS_REPORT.json",
        "KR_NEWS_NEW_INFORMATION_REPORT.json", "CANONICAL_RESEARCH_GATE_AUDIT.json",
        "RUN_STATUS.json", "V73_KR_NEWS_FOLD_AUDIT.csv",
        "V73_KR_NEWS_OUTER_EVIDENCE.csv.gz", "V73_DIAGNOSTIC_CHALLENGER_OOF.csv.gz",
        "V73_FAIL_CLOSED_SELECTED_OOF.csv.gz",
    )
    for name in runner_files:
        require(name in manifest.get("files", {}), f"V73 runner cache is unmanifested: {name}")
        atomic_bytes((CACHE_DIR / name).read_bytes(), out / name)

    atomic_json({
        "version": "V73", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
        "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
        "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
        "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
        "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
        "kr_news_outer": {
            "auc_delta": evaluation["outer_auc_delta"],
            "balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
            "confidence_correctness_auc_delta": evaluation["outer_confidence_correctness_auc_delta"],
        },
        "fallback_is_exact_v69": True, "seal_authorized": False,
    }, out / "SOURCE_TRANSFER_REPORT.json")
    atomic_json({
        "version": 73, "status": "VERIFIED_FULL_CACHE_ADOPTED",
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "cache_experiment_id": EXPECTED_CACHE_EXPERIMENT_ID,
        "all_manifested_files_verified": len(manifest.get("files", {})),
        "retraining_performed": False, "exact_v69_fallback_verified": True,
        "authority_audit": authority, "seal_authorized": False,
    }, out / "V73_CACHE_ADOPTION_REPORT.json")
    return {
        "status": evaluation["status"], "hypothesis": HYPOTHESIS,
        "material_pass": False, "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
        "outer_confidence_correctness_auc_delta": evaluation["outer_confidence_correctness_auc_delta"],
        "canonical_gate": recomputed_gate, "fallback_is_exact_v69": True,
        "output_written": True, "cache_adopted": True, "retraining_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=CONTROLLER_OUTPUT is None)
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if CONTROLLER_OUTPUT:
        if args.audit_only or args.smoke_test or args.full_run:
            parser.error("explicit modes cannot be combined with MARKET_BIO_VERSION_OUTPUT")
        args.full_run = True
        args.output = Path(CONTROLLER_OUTPUT)
    return args


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    authority = verify_authority()
    kr, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS, "authority": authority,
            "kr_news_dev_rows": len(kr), "kr_news_oof_rows": int(champion.source_family.eq("KR_NEWS").sum()),
            "canonical_controller_gate_checks": 14, "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    if CONTROLLER_OUTPUT and CACHE_DIR.is_dir():
        summary = adopt_verified_full_cache(authority, champion, args.output)
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    diagnostic, evidence, audits = run_nested(kr, champion, smoke=args.smoke_test)
    evaluation = evaluate(champion, diagnostic, evidence, audits, 250 if args.smoke_test else BOOTSTRAP_DRAWS)
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "material_pass": evaluation["material_pass"],
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
        "outer_confidence_correctness_auc_delta": evaluation["outer_confidence_correctness_auc_delta"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "fallback_is_exact_v69": not evaluation["material_pass"], "output_written": False,
    }
    if args.smoke_test:
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    controller_result = write_outputs(authority, champion, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": controller_result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
