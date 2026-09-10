"""V225 strict-OOF high-confidence news correctness discriminator.

Direction is the immutable V69 outer-OOF prediction.  This experiment predicts
whether that direction is correct; it never predicts UP/DOWN itself.  All
models are source-specific, chronologically cross-fitted with a 35-minute
embargo, and publisher/topic reliability features are computed from past
training outcomes only.  Research Seal, Final Meta and DEV_EXTENSION are not
read.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import experiment_v37_text as v37
import runtime_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "research" / "v224_news_confidence" / "V225_STRICT_OOF"
DEV = DATA / "dev_contract_v36_labeled.csv.gz"
EVENTS = DATA / "dev_contract_v36_events.csv.gz"
V69 = ROOT / "output_V69" / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
V69_COMMIT = ROOT / "output_V69" / "COMMIT.json"
V69_MANIFEST = ROOT / "output_V69" / "ARTIFACT_MANIFEST.json"
EXPECTED_DEV_SHA = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_SHA = "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002"
EXPECTED_V69_COMMIT_SHA = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EMBARGO = pd.Timedelta(minutes=35)
OUTER_BLOCKS = 6
INNER_FRACTION = 0.70
HC_QUANTILE = 0.80
COST = 0.002
SEED = 22501

EXPERT_PROBS = (
    "prob", "prob_opportunity_direction", "prob_event_form_direction",
    "prob_fixed_numeric_direction",
)
EXPERT_CONFIDENCE = (
    "confidence_signal", "opportunity_confidence", "confidence_opportunity_direction",
    "confidence_event_form_direction", "confidence_fixed_numeric_direction",
)
MARKET_COLUMNS = (
    "pre_ret_2", "pre_ret_5", "pre_ret_15", "pre_ret_30", "pre_ret_60",
    "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
    "volume_ratio_1_30", "volume_ratio_5_30", "volume_ratio_10_60",
    "range_5m", "range_10m", "range_30m", "range_60m",
    "close_position_10", "close_position_30", "entry_bar_ret", "entry_bar_range",
    "intraday_ret_open", "return_autocorr_30", "up_fraction_10", "up_fraction_30",
    "trend_slope_30", "trend_slope_60", "vwap_distance_30", "minutes_from_open",
    "minutes_to_close",
)
PUBLISHER_CLASSES = (
    "WIRE", "FINANCIAL_PRESS", "MAJOR_PRESS", "BIOTECH_SPECIALIST",
    "ISSUER_PR", "PR_DISTRIBUTION", "AGGREGATOR", "OTHER",
)
TOPICS = {
    "CLINICAL_TRIAL": r"\b(?:phase\s*[123iIvV]+|clinical trial|top-?line|endpoint|patient|study)\b",
    "FDA_REGULATORY": r"\b(?:fda|regulator|approval|approved|crl|complete response|clinical hold|designation)\b",
    "FINANCING": r"\b(?:offering|financing|registered direct|at-the-market|convertible|warrant|private placement)\b",
    "MA": r"\b(?:merger|acquisition|acquire[sd]?|takeover|buyout)\b",
    "LICENSING": r"\b(?:licen[cs]|collaboration|partnership|royalt|milestone payment)\b",
    "EARNINGS": r"\b(?:earnings|quarter|revenue|guidance|profit|loss|results of operations)\b",
    "SAFETY": r"\b(?:adverse event|safety signal|recall|toxicity|death)\b",
    "MANAGEMENT": r"\b(?:appoint|resign|chief executive|ceo|cfo|director|management)\b",
    "MANUFACTURING": r"\b(?:manufactur|facility|cgmp|supply|production)\b",
    "PATENT": r"\b(?:patent|intellectual property|infringement)\b",
}
POSITIVE = re.compile(r"(?i)\b(?:positive|met endpoint|approved|approval|breakthrough|success|beat|surpass)\b")
NEGATIVE = re.compile(r"(?i)\b(?:negative|missed endpoint|failed|failure|crl|clinical hold|recall|terminated)\b")

FAMILIES = {
    "NATIVE": ("native", False),
    "AGREEMENT": ("agreement", False),
    "PUBLISHER": ("publisher", False),
    "TOPIC": ("topic", False),
    "CONSENSUS_NOVELTY": ("consensus", False),
    "MARKET_CONTEXT": ("market", False),
    "FULL_STRUCTURED": ("full", False),
    "FULL_STRUCTURED_TEXT": ("full", True),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            compressed.write(raw)
    os.replace(temporary, path)


def verify_authority() -> dict[str, Any]:
    checks = {
        str(DEV.relative_to(ROOT)): (sha256(DEV), EXPECTED_DEV_SHA),
        str(V69.relative_to(ROOT)): (sha256(V69), EXPECTED_V69_SHA),
        str(V69_COMMIT.relative_to(ROOT)): (sha256(V69_COMMIT), EXPECTED_V69_COMMIT_SHA),
        str(V69_MANIFEST.relative_to(ROOT)): (sha256(V69_MANIFEST), EXPECTED_V69_MANIFEST_SHA),
    }
    failures = {name: {"actual": actual, "expected": expected}
                for name, (actual, expected) in checks.items() if actual != expected}
    if failures:
        raise RuntimeError(f"V225_AUTHORITY_HASH_FAIL:{failures}")
    return {
        "hashes": {name: actual for name, (actual, _) in checks.items()},
        "authorized_labels": "immutable V36 DEV only",
        "direction": "immutable V69 outer-OOF only",
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "research_seal_loaded": False,
        "final_meta_loaded": False,
    }


def domain(value: Any) -> str:
    try:
        host = urlparse(str(value or "")).netloc.lower().split(":", 1)[0]
    except Exception:
        return "<MISSING>"
    return host[4:] if host.startswith("www.") else (host or "<MISSING>")


def publisher_class(row: pd.Series) -> str:
    value = f"{row.get('publisher_domain','')} {row.get('source','')}".lower()
    if "reuters" in value or "apnews" in value or "associated press" in value:
        return "WIRE"
    if any(token in value for token in ("bloomberg", "marketwatch", "wsj", "cnbc", "ft.com", "financialtimes")):
        return "FINANCIAL_PRESS"
    if any(token in value for token in ("nytimes", "washingtonpost", "cnn", "bbc", "forbes")):
        return "MAJOR_PRESS"
    if any(token in value for token in ("biospace", "fiercebiotech", "endpts", "statnews", "biopharmadive")):
        return "BIOTECH_SPECIALIST"
    if any(token in value for token in ("globenewswire", "prnewswire", "businesswire", "accesswire")):
        return "PR_DISTRIBUTION"
    if any(token in value for token in ("investor", "ir.", "company release", "issuer")):
        return "ISSUER_PR"
    if any(token in value for token in ("news.google", "naver", "yahoo", "bing")):
        return "AGGREGATOR"
    return "OTHER"


def topic_flags(text: str) -> dict[str, float]:
    value = str(text or "")
    flags = {name: float(bool(re.search(pattern, value, re.I))) for name, pattern in TOPICS.items()}
    if not any(flags.values()):
        flags["OTHER"] = 1.0
    else:
        flags["OTHER"] = 0.0
    return flags


def normalized_tokens(value: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9가-힣]{3,}", str(value or "").lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def add_causal_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    data["publisher_domain"] = data.url.map(domain)
    data["publisher_class"] = data.apply(publisher_class, axis=1)
    combined_text = (data.headline.fillna("").astype(str) + " " + data.body.fillna("").astype(str)).str[:16000]
    flags = pd.DataFrame([topic_flags(value) for value in combined_text], index=data.index)
    for column in flags:
        data[f"topic__{column}"] = flags[column].astype(float)
    topic_columns = [column for column in data if column.startswith("topic__")]
    data["topic_primary"] = flags.idxmax(axis=1).str.replace("topic__", "", regex=False)
    data["publisher_topic"] = data.publisher_class + "|" + data.topic_primary
    data["positive_semantic"] = combined_text.map(lambda value: float(bool(POSITIVE.search(value))))
    data["negative_semantic"] = combined_text.map(lambda value: float(bool(NEGATIVE.search(value))))
    data["headline_tokens"] = data.headline.map(normalized_tokens)
    for column in (
        "decision_news_rows", "decision_source_clusters", "decision_publisher_entropy",
        "decision_source_type_diversity", "decision_mean_similarity", "decision_min_similarity",
        "decision_topic_agreement", "decision_contradiction", "structured_confirmation",
        "novelty_count_24h", "novelty_count_7d", "novelty_count_30d", "novelty_max_similarity_7d",
    ):
        data[column] = 0.0
    # Only immutable V69 OOF rows participate.  For a row at t, consensus may
    # use publication timestamps through t+2m; novelty uses strictly earlier rows.
    for _, indices in data.groupby("ticker", sort=False).groups.items():
        positions = np.asarray(sorted(indices), dtype=int)
        group = data.loc[positions]
        times = pd.to_datetime(group.event_time_utc, utc=True)
        for position in positions:
            timestamp = pd.Timestamp(data.at[position, "event_time_utc"])
            decision = positions[(times >= timestamp - pd.Timedelta(minutes=30)).to_numpy()
                                 & (times <= timestamp + pd.Timedelta(minutes=2)).to_numpy()]
            news = [idx for idx in decision if data.at[idx, "source_family"] in {"US_NEWS", "KR_NEWS"}]
            domains = [str(data.at[idx, "publisher_domain"]) for idx in news]
            classes = [str(data.at[idx, "publisher_class"]) for idx in news]
            counts = pd.Series(domains).value_counts() if domains else pd.Series(dtype=int)
            probabilities = counts.to_numpy(float) / counts.sum() if len(counts) else np.asarray([])
            similarities = []
            for left_index, left in enumerate(news):
                for right in news[left_index + 1:]:
                    similarities.append(jaccard(data.at[left, "headline_tokens"], data.at[right, "headline_tokens"]))
            topics = [str(data.at[idx, "topic_primary"]) for idx in news]
            data.at[position, "decision_news_rows"] = len(news)
            data.at[position, "decision_source_clusters"] = len(set(domains))
            data.at[position, "decision_publisher_entropy"] = float(
                -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum()
            ) if len(probabilities) else 0.0
            data.at[position, "decision_source_type_diversity"] = len(set(classes))
            data.at[position, "decision_mean_similarity"] = float(np.mean(similarities)) if similarities else 1.0
            data.at[position, "decision_min_similarity"] = float(np.min(similarities)) if similarities else 1.0
            data.at[position, "decision_topic_agreement"] = (
                max(pd.Series(topics).value_counts()) / len(topics) if topics else 1.0
            )
            positive = any(bool(data.at[idx, "positive_semantic"]) for idx in news)
            negative = any(bool(data.at[idx, "negative_semantic"]) for idx in news)
            data.at[position, "decision_contradiction"] = float(positive and negative)
            data.at[position, "structured_confirmation"] = float(any(
                data.at[idx, "source_family"] in {"US_SEC", "KR_KIND"} for idx in decision
            ))
            past = positions[(times < timestamp).to_numpy()]
            for label, window in (("24h", 1), ("7d", 7), ("30d", 30)):
                count = sum(pd.Timestamp(data.at[idx, "event_time_utc"]) >= timestamp - pd.Timedelta(days=window)
                            for idx in past)
                data.at[position, f"novelty_count_{label}"] = count
            past7 = [idx for idx in past if pd.Timestamp(data.at[idx, "event_time_utc"])
                     >= timestamp - pd.Timedelta(days=7)]
            data.at[position, "novelty_max_similarity_7d"] = max(
                [jaccard(data.at[position, "headline_tokens"], data.at[idx, "headline_tokens"])
                 for idx in past7], default=0.0,
            )
    data["news_text"] = (data.headline.fillna("").astype(str) + " " + data.body.fillna("").astype(str)).str[:12000]
    return data


def load_frame() -> pd.DataFrame:
    pred = pd.read_csv(V69, compression="gzip", low_memory=False, dtype={"ticker": str},
                       parse_dates=["event_time_utc"])
    dev = pd.read_csv(DEV, compression="gzip", low_memory=False, dtype={"ticker": str},
                      parse_dates=["event_time_utc"])
    aligned = dev.set_index("event_id").loc[pred.event_id].reset_index()
    if not np.array_equal(aligned.y.to_numpy(int), pred.y.to_numpy(int)):
        raise RuntimeError("V225_V36_V69_LABEL_MISMATCH")
    events = pd.read_csv(EVENTS, compression="gzip", low_memory=False,
                         usecols=["event_id", "url", "article_id", "timestamp_quality"])
    events = events.drop_duplicates("event_id")
    frame = pred.copy()
    for column in ("source", "event_type", "headline", "body", *MARKET_COLUMNS,
                   "event_positive_kw", "event_negative_kw", "event_financing_kw",
                   "event_trial_kw", "event_regulatory_kw", "event_ma_kw", "event_earnings_kw"):
        frame[column] = aligned[column].to_numpy()
    frame = frame.merge(events, on="event_id", how="left", validate="one_to_one")
    frame["headline"] = frame.headline.fillna("").astype(str)
    frame["body"] = frame.body.fillna("").astype(str)
    # normalized_cluster treats a truthy float NaN as an external identifier;
    # canonicalize missing metadata before constructing the label-blind key.
    frame["url"] = frame.url.fillna("").astype(str)
    frame["article_id"] = frame.article_id.fillna("").astype(str)
    frame["text_cluster_id"] = frame.apply(v37.normalized_cluster, axis=1)
    frame["cluster_weight"] = 1.0 / frame.groupby("text_cluster_id").event_id.transform("size")
    frame["direction_prediction"] = frame.prob >= 0.5
    frame["correct_target"] = (frame.direction_prediction == frame.y.astype(bool)).astype(int)
    frame["signed_net"] = np.where(frame.direction_prediction, 1.0, -1.0) * frame.fwd_ret_30m - COST
    frame["material_target"] = (frame.fwd_ret_30m.abs() >= 0.005).astype(int)
    frame = add_causal_metadata(frame)
    news = frame[frame.source_family.isin(["US_NEWS", "KR_NEWS"])].copy()
    if news.event_id.duplicated().any() or not news.correct_target.nunique() == 2:
        raise RuntimeError("V225_NEWS_ID_OR_TARGET_FAIL")
    return news


def folds(frame: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    ordered = frame.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    blocks = np.array_split(np.arange(len(ordered)), OUTER_BLOCKS)
    output = []
    for fold in range(1, OUTER_BLOCKS):
        valid = ordered.iloc[blocks[fold]].copy()
        boundary = valid.event_time_utc.min()
        train = ordered[ordered.event_time_utc < boundary - EMBARGO].copy()
        train = train[~train.text_cluster_id.isin(set(valid.text_cluster_id))].copy()
        if len(train) < 120 or len(valid) < 100 or train.correct_target.nunique() != 2:
            raise RuntimeError(f"V225_OUTER_FOLD_FAIL:{fold}")
        output.append((train, valid, fold))
    return output


def inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    split = max(90, int(INNER_FRACTION * len(ordered)))
    valid = ordered.iloc[split:].copy()
    boundary = valid.event_time_utc.min()
    base = ordered[ordered.event_time_utc < boundary - EMBARGO].copy()
    base = base[~base.text_cluster_id.isin(set(valid.text_cluster_id))].copy()
    if len(base) < 70 or len(valid) < 30 or base.correct_target.nunique() != 2:
        raise RuntimeError("V225_INNER_SPLIT_FAIL")
    return base, valid


def causal_train_encoding(frame: pd.DataFrame, column: str, smoothing: float = 16.0) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["event_time_utc", "event_id"], kind="stable")
    total_n = 0.0
    total_y = 0.0
    counts: defaultdict[str, float] = defaultdict(float)
    positives: defaultdict[str, float] = defaultdict(float)
    values: dict[int, float] = {}
    supports: dict[int, float] = {}
    for index, row in ordered.iterrows():
        key = str(row.get(column, "<MISSING>"))
        global_rate = total_y / total_n if total_n else 0.5
        values[index] = (positives[key] + smoothing * global_rate) / (counts[key] + smoothing)
        supports[index] = math.log1p(counts[key])
        weight = float(row.cluster_weight)
        counts[key] += weight
        positives[key] += weight * int(row.correct_target)
        total_n += weight
        total_y += weight * int(row.correct_target)
    return (np.asarray([values[index] for index in frame.index], float),
            np.asarray([supports[index] for index in frame.index], float))


def target_encoding(train: pd.DataFrame, target: pd.DataFrame, column: str,
                    smoothing: float = 16.0) -> tuple[np.ndarray, np.ndarray]:
    weights = train.cluster_weight.to_numpy(float)
    global_rate = float(np.average(train.correct_target, weights=weights))
    work = pd.DataFrame({"key": train[column].fillna("<MISSING>").astype(str),
                         "weight": weights,
                         "positive": weights * train.correct_target.to_numpy(int)})
    stats = work.groupby("key").agg(weight=("weight", "sum"), positive=("positive", "sum"))
    reliability = (stats.positive + smoothing * global_rate) / (stats.weight + smoothing)
    keys = target[column].fillna("<MISSING>").astype(str)
    return (keys.map(reliability).fillna(global_rate).to_numpy(float),
            np.log1p(keys.map(stats.weight).fillna(0.0).to_numpy(float)))


def base_features(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["direction_margin"] = np.abs(frame.prob.to_numpy(float) - 0.5)
    output["predicted_up"] = frame.direction_prediction.astype(float)
    for column in EXPERT_CONFIDENCE:
        output[f"confidence__{column}"] = pd.to_numeric(frame[column], errors="coerce")
    if family in {"agreement", "full"}:
        direction = frame.prob.to_numpy(float) >= 0.5
        agreements = []
        aligned = []
        for column in EXPERT_PROBS:
            probability = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            candidate_direction = probability >= 0.5
            output[f"expert_margin__{column}"] = np.abs(probability - 0.5)
            output[f"expert_agree__{column}"] = (candidate_direction == direction).astype(float)
            agreements.append((candidate_direction == direction).astype(float))
            aligned.append(np.where(direction, probability, 1.0 - probability))
        output["agreement_fraction"] = np.column_stack(agreements).mean(axis=1)
        output["agreement_dispersion"] = np.column_stack(aligned).std(axis=1)
    if family in {"publisher", "full"}:
        for name in PUBLISHER_CLASSES:
            output[f"publisher_class__{name}"] = frame.publisher_class.eq(name).astype(float)
        output["timestamp_exact"] = frame.timestamp_quality.fillna("").astype(str).str.upper().eq("EXACT").astype(float)
    if family in {"topic", "full"}:
        for column in [column for column in frame if column.startswith("topic__")]:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
        output["positive_semantic"] = frame.positive_semantic
        output["negative_semantic"] = frame.negative_semantic
    if family in {"consensus", "full"}:
        for column in (
            "decision_news_rows", "decision_source_clusters", "decision_publisher_entropy",
            "decision_source_type_diversity", "decision_mean_similarity", "decision_min_similarity",
            "decision_topic_agreement", "decision_contradiction", "structured_confirmation",
            "novelty_count_24h", "novelty_count_7d", "novelty_count_30d", "novelty_max_similarity_7d",
        ):
            output[column] = pd.to_numeric(frame[column], errors="coerce")
    if family in {"market", "full"}:
        for column in MARKET_COLUMNS:
            output[f"market__{column}"] = pd.to_numeric(frame[column], errors="coerce")
    return output.replace([np.inf, -np.inf], np.nan)


def feature_pair(train: pd.DataFrame, target: pd.DataFrame, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = base_features(train, family)
    x_target = base_features(target, family)
    reliability_columns = []
    if family in {"publisher", "full"}:
        reliability_columns.extend(["publisher_domain", "publisher_class"])
    if family in {"topic", "full"}:
        reliability_columns.extend(["topic_primary", "publisher_topic"])
    for column in reliability_columns:
        train_rel, train_support = causal_train_encoding(train, column)
        target_rel, target_support = target_encoding(train, target, column)
        x_train[f"past_correctness__{column}"] = train_rel
        x_train[f"past_support__{column}"] = train_support
        x_target[f"past_correctness__{column}"] = target_rel
        x_target[f"past_support__{column}"] = target_support
    return x_train, x_target


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, family: str, use_text: bool,
                seed: int) -> np.ndarray:
    x_train, x_target = feature_pair(train, target, family)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(imputer.fit_transform(x_train))
    target_numeric = scaler.transform(imputer.transform(x_target))
    if use_text:
        vector = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                                 min_df=3, max_df=0.995, max_features=24000,
                                 sublinear_tf=True, dtype=np.float32)
        train_text = vector.fit_transform(train.news_text)
        target_text = vector.transform(target.news_text)
        train_matrix = sparse.hstack([sparse.csr_matrix(train_numeric), train_text], format="csr")
        target_matrix = sparse.hstack([sparse.csr_matrix(target_numeric), target_text], format="csr")
    else:
        train_matrix, target_matrix = train_numeric, target_numeric
    model = LogisticRegression(C=0.25, solver="liblinear", class_weight="balanced",
                               max_iter=1000, random_state=seed)
    model.fit(train_matrix, train.correct_target.to_numpy(int),
              sample_weight=train.cluster_weight.to_numpy(float))
    return np.clip(model.predict_proba(target_matrix)[:, 1], 1e-6, 1 - 1e-6)


def run_family(frame: pd.DataFrame, family_name: str, feature_family: str,
               use_text: bool) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    pieces = []
    audits = []
    for train, valid, fold in folds(frame):
        inner_train, inner_valid = inner_split(train)
        inner_probability = fit_predict(inner_train, inner_valid, feature_family, use_text,
                                        SEED + fold * 100)
        threshold = float(np.quantile(inner_probability, HC_QUANTILE))
        probability = fit_predict(train, valid, feature_family, use_text, SEED + fold * 100 + 1)
        output = valid[["event_id", "event_time_utc", "ticker", "source_family", "fold", "y",
                        "fwd_ret_30m", "prob", "direction_prediction", "correct_target",
                        "signed_net", "cluster_weight", "text_cluster_id", "publisher_class",
                        "topic_primary"]].copy()
        output["p_correct_news"] = probability
        output["news_high_conf"] = probability >= threshold
        output["meta_fold"] = fold
        output["candidate"] = family_name
        pieces.append(output)
        audits.append({
            "fold": fold,
            "train_n": len(train),
            "inner_train_n": len(inner_train),
            "inner_valid_n": len(inner_valid),
            "valid_n": len(valid),
            "train_end": train.event_time_utc.max(),
            "valid_start": valid.event_time_utc.min(),
            "embargo_minutes": 35,
            "text_cluster_overlap": len(set(train.text_cluster_id) & set(valid.text_cluster_id)),
            "threshold_source": "inner validation score distribution only",
            "p_correct_threshold": threshold,
            "outer_labels_used_for_fit_or_threshold": False,
        })
        print(f"[V225] {frame.source_family.iloc[0]} {family_name} fold={fold}/5 "
              f"train={len(train)} valid={len(valid)}", flush=True)
    return pd.concat(pieces, ignore_index=True), audits


def ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(target)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right if right < 1 else probability <= right)
        if mask.any():
            value += mask.mean() * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def bootstrap_net(frame: pd.DataFrame, draws: int = 1000) -> dict[str, float | int]:
    work = frame.copy().reset_index(drop=True)
    work["day"] = pd.to_datetime(work.event_time_utc, utc=True).dt.date
    groups = {day: np.asarray(index, int) for day, index in work.groupby("day").indices.items()}
    days = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(SEED + 99)
    values = []
    for _ in range(draws):
        positions = np.concatenate([groups[day] for day in rng.choice(days, len(days), replace=True)])
        sample = work.iloc[positions]
        selected = sample.news_high_conf.to_numpy(bool)
        if selected.any():
            values.append(float(sample.loc[selected, "signed_net"].mean()))
    return {"draws": len(values), "lower95": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)), "upper95": float(np.quantile(values, 0.975))}


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame.correct_target.to_numpy(int)
    probability = frame.p_correct_news.to_numpy(float)
    selected = frame.news_high_conf.to_numpy(bool)
    net = frame.loc[selected, "signed_net"].to_numpy(float)
    ordered_net = np.sort(net)
    ranked = pd.qcut(pd.Series(probability).rank(method="first"), 5, labels=False)
    bins = pd.DataFrame({"bin": ranked, "correct": target, "p": probability}).groupby("bin").agg(
        n=("correct", "size"), accuracy=("correct", "mean"), mean_p=("p", "mean")
    )
    risk_coverage = {}
    for coverage in (0.15, 0.20, 0.30, 0.50):
        cutoff = float(np.quantile(probability, 1.0 - coverage))
        mask = probability >= cutoff
        risk_coverage[str(coverage)] = {"n": int(mask.sum()), "coverage": float(mask.mean()),
                                        "accuracy": float(target[mask].mean()),
                                        "mean_net": float(frame.loc[mask, "signed_net"].mean())}
    return {
        "n": len(frame),
        "correct_rate": float(target.mean()),
        "confidence_correctness_auc": float(roc_auc_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "ece_10": ece(target, probability),
        "hc_n": int(selected.sum()),
        "hc_coverage": float(selected.mean()),
        "hc_accuracy": float(target[selected].mean()),
        "hc_net": float(net.mean()),
        "hc_winsorized_net": float(np.clip(net, *np.quantile(net, [0.01, 0.99])).mean()),
        "hc_top1_removed_net": float(ordered_net[:-1].mean()) if len(net) > 1 else None,
        "hc_top5_removed_net": float(ordered_net[:-5].mean()) if len(net) > 5 else None,
        "bootstrap_hc_net": bootstrap_net(frame),
        "confidence_bins": {str(int(index)): row.to_dict() for index, row in bins.iterrows()},
        "confidence_bin_non_decreasing_fraction": float(
            np.mean(np.diff(bins.accuracy.to_numpy(float)) >= 0)
        ),
        "risk_coverage": risk_coverage,
    }


def native_summary(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame.correct_target.to_numpy(int)
    probability = pd.to_numeric(frame.confidence_signal, errors="coerce").fillna(0).to_numpy(float)
    high = frame.high_conf.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(bool)
    return {
        "n": len(frame),
        "confidence_correctness_auc": float(roc_auc_score(target, probability)),
        "hc_n": int(high.sum()), "hc_coverage": float(high.mean()),
        "hc_accuracy": float(target[high].mean()),
        "hc_net": float(frame.loc[high, "signed_net"].mean()),
    }


def main() -> int:
    runtime_limits.configure()
    authority = verify_authority()
    news = load_frame()
    OUT.mkdir(parents=True, exist_ok=True)
    source_reports = {}
    all_predictions = []
    all_audits = []
    for source_family in ("US_NEWS", "KR_NEWS"):
        source = news[news.source_family.eq(source_family)].copy()
        candidate_reports = {}
        source_predictions = {}
        source_audits = {}
        for family_name, (feature_family, use_text) in FAMILIES.items():
            predictions, audits = run_family(source, family_name, feature_family, use_text)
            candidate_reports[family_name] = summarize(predictions)
            source_predictions[family_name] = predictions
            source_audits[family_name] = audits
            all_predictions.append(predictions)
            all_audits.extend({"source_family": source_family, "candidate": family_name, **row}
                              for row in audits)
        ranked = sorted(candidate_reports, key=lambda name: (
            candidate_reports[name]["confidence_correctness_auc"],
            candidate_reports[name]["hc_accuracy"],
            candidate_reports[name]["hc_net"],
        ), reverse=True)
        best = ranked[0]
        source_reports[source_family] = {
            "inventory_n": len(source),
            "native_v69_confidence": native_summary(source),
            "candidates": candidate_reports,
            "dev_selected_candidate": best,
            "selection_bias": "candidate family selected on current DEV OOF; independent confirmation required",
        }
        print("\n========= HIGH-CONFIDENCE NEWS RESEARCH =========", flush=True)
        print(f"{source_family} N={len(source)} best={best}", flush=True)
        print(json.dumps(candidate_reports[best], ensure_ascii=False, indent=2)[:5000], flush=True)
    report = {
        "version": 225,
        "hypothesis": "SOURCE_SPECIFIC_STRICT_OOF_NEWS_CORRECTNESS_DISCRIMINATOR",
        "status": "DEV_RESEARCH_ONLY",
        "authority": authority,
        "contract": {
            "target": "P_CORRECT_NEWS = P(immutable V69 outer-OOF direction equals y)",
            "same_row_fitted_direction_used": False,
            "source_specific": ["US_NEWS", "KR_NEWS"],
            "outer_folds": 5,
            "embargo_minutes": 35,
            "hc_threshold": "fixed 80th percentile of inner-past score distribution",
            "publisher_reliability": "Bayesian-smoothed past-train outcomes only",
            "topic_reliability": "Bayesian-smoothed past-train outcomes only",
            "consensus_cutoff": "publication timestamp <= event t+2m only",
            "direction_changed": False,
        },
        "sources": source_reports,
        "experiment_budget_used": len(FAMILIES),
        "experiment_budget_limit": 15,
        "seal_authorized": False,
        "final_meta_authorized": False,
        "research_seal_or_final_rows_read": False,
    }
    atomic_json(report, OUT / "V225_HIGH_CONFIDENCE_NEWS_REPORT.json")
    atomic_csv_gz(pd.concat(all_predictions, ignore_index=True), OUT / "V225_ALL_CANDIDATE_OOF.csv.gz")
    atomic_csv_gz(pd.DataFrame(all_audits), OUT / "V225_NESTED_FOLD_AUDIT.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "MANIFEST.json"}
    atomic_json({"version": 225, "files": files, "runner_sha256": sha256(Path(__file__)),
                 "research_seal_or_final_rows_read": False}, OUT / "MANIFEST.json")
    print(json.dumps({source: {
        "best": value["dev_selected_candidate"],
        "metrics": value["candidates"][value["dev_selected_candidate"]],
    } for source, value in source_reports.items()}, ensure_ascii=False, indent=2)[:12000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
