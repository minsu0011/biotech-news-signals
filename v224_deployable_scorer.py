"""Standalone runtime for the frozen V224 prospective-confirmation scorer.

This module contains inference-time preprocessing and the causal V224 state
machine.  It never trains, opens labels while scoring, or reads Seal/Final.
The fitted estimators live in FROZEN_V224_DEPLOYABLE_SCORER.pkl.gz.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


SCHEMA_VERSION = 1
MODEL_FAMILY = "V224_EXACT_SEC_N_T_S_GUARDED"
POLARITY_AUC_FLOOR = 0.48
CUTOFF_GRID = np.arange(0.30, 0.701, 0.01)
CONFIDENCE_QUANTILE = 0.80
DIRECTION_THRESHOLD = 0.50
COST = 0.002
CONFIRMATION_BLOCK_SIZE = 20
BLEND_WEIGHTS = {"N": 0.50, "T": 0.25, "S": 0.25}
PROBABILITY_CLIP = 1e-5
CAT_COLUMNS = ("source_family", "event_type", "form")
FORBIDDEN_SCORE_COLUMNS = {
    "y", "fwd_ret_30m", "future_return", "correct", "correctness",
    "strategy_net", "model_correctness", "up_down",
}

NUMERIC_COLS = [
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
    "excess_ret_5", "excess_ret_15", "excess_ret_30", "excess_ret_60", "event_hour",
    "event_weekday", "event_month", "event_hour_sin", "event_hour_cos", "abs_pre_ret_2",
    "abs_pre_ret_5", "abs_pre_ret_15", "abs_pre_ret_30", "abs_pre_ret_60",
    "ret_slope_2_15", "ret_slope_5_30", "kw_sentiment",
]

ITEM_RE = re.compile(r"(?i)\bitem\s+(1\.01|2\.02|2\.03|3\.02|5\.02|7\.01|8\.01|9\.01)\b")
EX99_RE = re.compile(r"(?mi)^\s*EX(?:HIBIT)?[-_ ]?99(?:\.\d+)?\s*$")
SIGNATURE_RE = re.compile(r"(?mi)^\s*SIGNATURES?\s*$")
DISCLAIMER_RE = re.compile(r"(?is)(forward[- ]looking statements?|safe harbor).{0,5000}$")
TAXONOMY = {
    "fda_approval": r"\bfda\b.{0,60}\bapprov(?:al|ed)\b|\bapproval\b.{0,60}\bfda\b",
    "crl": r"complete response letter|\bcrl\b|refusal to file",
    "clinical_hold_imposed": r"clinical hold.{0,80}(?:impos|placed|initiated)|(?:impos|placed).{0,80}clinical hold",
    "clinical_hold_lifted": r"clinical hold.{0,80}(?:lift|remov)|(?:lift|remov).{0,80}clinical hold",
    "phase_1": r"\bphase\s*(?:1|i)\b", "phase_2": r"\bphase\s*(?:2|ii)\b",
    "phase_3": r"\bphase\s*(?:3|iii)\b",
    "endpoint_met": r"(?:met|achieved)\s+(?:the\s+)?(?:primary|secondary)?\s*endpoint",
    "endpoint_missed": r"(?:missed|did not meet|failed to meet)\s+(?:the\s+)?(?:primary|secondary)?\s*endpoint",
    "trial_positive": r"positive (?:topline|top-line|clinical|trial) results?|statistically significant",
    "trial_negative": r"negative (?:topline|top-line|clinical|trial) results?|not statistically significant",
    "public_offering": r"public offering", "registered_direct": r"registered direct",
    "atm": r"at-the-market|\batm (?:program|facility|offering)",
    "convertible": r"convertible (?:note|debt|security)", "warrant": r"\bwarrants?\b",
    "offering_priced": r"offering.{0,100}\bpriced\b|priced.{0,100}offering",
    "offering_closed": r"offering.{0,100}\bclos(?:e|ed|ing)\b|clos(?:e|ed|ing).{0,100}offering",
    "license": r"licens(?:e|ing|ed) agreement", "partnership": r"partnership|collaboration agreement",
    "upfront": r"upfront payment", "milestone": r"milestone payment",
    "ma_definitive": r"definitive (?:merger|acquisition) agreement|agreement and plan of merger",
    "ma_terminated": r"terminat(?:e|ed|ion).{0,100}(?:merger|acquisition) agreement",
    "earnings": r"earnings|results of operations|quarterly results",
    "guidance": r"financial guidance|revis(?:e|ed) guidance",
    "management_appointment": r"appoint(?:ed|ment).{0,100}(?:chief|director|officer|president)",
    "management_resignation": r"resign(?:ed|ation).{0,100}(?:chief|director|officer|president)",
    "safety": r"serious adverse event|safety signal", "recall": r"product recall|voluntary recall",
    "manufacturing": r"manufacturing (?:issue|deficien|facility)|cGMP deficien",
}
AMOUNT_RE = re.compile(r"(?i)(?:\$|usd\s*)(\d+(?:\.\d+)?)\s*(million|billion|m|bn)?")
OFFER_PRICE_RE = re.compile(r"(?i)(?:offering|purchase) price.{0,80}?\$\s*(\d+(?:\.\d+)?)")
TRIAL_N_RE = re.compile(r"(?i)\b(?:n\s*=|enroll(?:ed|ment of))\s*(\d{1,5})\b")
PVALUE_RE = re.compile(r"(?i)\bp\s*[=<]\s*(0?\.\d+)")
HR_RE = re.compile(r"(?i)hazard ratio.{0,30}?(0?\.\d+)")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, default=str) + "\n").encode("utf-8")


def numeric_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact inference equivalent of bio_news_30m_v3.model_frame + V36 categories."""
    x = frame.copy()
    for column in NUMERIC_COLS:
        if column not in x.columns:
            x[column] = np.nan
    for column in CAT_COLUMNS:
        if column not in x.columns:
            x[column] = ""
    if "event_time_utc" not in x.columns:
        raise ValueError("V224_MISSING_EVENT_TIME_UTC")
    ts = pd.to_datetime(x["event_time_utc"], utc=True, errors="raise")
    x["event_hour"] = ts.dt.hour + ts.dt.minute / 60.0
    x["event_weekday"] = ts.dt.weekday
    x["event_month"] = ts.dt.month
    x["event_hour_sin"] = np.sin(2.0 * np.pi * x["event_hour"] / 24.0)
    x["event_hour_cos"] = np.cos(2.0 * np.pi * x["event_hour"] / 24.0)
    for suffix in ("2", "5", "15", "30", "60"):
        x[f"abs_pre_ret_{suffix}"] = pd.to_numeric(x[f"pre_ret_{suffix}"], errors="coerce").abs()
    x["ret_slope_2_15"] = x["pre_ret_2"] - x["pre_ret_15"]
    x["ret_slope_5_30"] = x["pre_ret_5"] - x["pre_ret_30"]
    x["kw_sentiment"] = x["event_positive_kw"] - x["event_negative_kw"] - x["event_financing_kw"]
    for suffix in ("5", "15", "30", "60"):
        x[f"excess_ret_{suffix}"] = (
            pd.to_numeric(x[f"pre_ret_{suffix}"], errors="coerce")
            - pd.to_numeric(x[f"benchmark_ret_{suffix}"], errors="coerce")
        )
    output = x[NUMERIC_COLS].copy().reset_index(drop=True)
    for column in CAT_COLUMNS:
        output[column] = x[column].fillna("").astype(str).reset_index(drop=True)
    return output


def segments(row: pd.Series) -> dict[str, str]:
    headline = str(row.get("headline", "") or "")
    body = str(row.get("body", "") or "")
    body = DISCLAIMER_RE.sub(" ", body)
    signature = SIGNATURE_RE.search(body)
    if signature:
        body = body[:signature.start()]
    matches = list(ITEM_RE.finditer(body))
    item_parts = []
    for number, match in enumerate(matches[:20]):
        end = matches[number + 1].start() if number + 1 < len(matches) else len(body)
        item_parts.append(body[match.start():end])
    exmatch = EX99_RE.search(body)
    ex99 = body[exmatch.end():] if exmatch else ""
    primary = " ".join(item_parts) if item_parts else body[:12000]
    if exmatch:
        primary = primary[:max(0, primary.find(exmatch.group(0)))]
    return {
        "headline": headline,
        "primary": primary[:16000],
        "ex99": ex99[:16000],
        "combined": f"HEADLINE {headline} PRIMARY {primary[:12000]} EX99 {ex99[:12000]}",
    }


def structured_features(row: pd.Series) -> dict[str, float]:
    part = segments(row)
    body = str(row.get("body", "") or "")
    text = (part["headline"] + " " + part["primary"] + " " + part["ex99"]).lower()
    result: dict[str, float] = {
        f"form={str(row.get('form', '')).upper()}": 1.0,
        f"event={str(row.get('event_type', '')).upper()}": 1.0,
        "body_present": float(bool(body.strip())),
        "ex99_present": float(bool(part["ex99"])),
        "log_body_len": float(np.log1p(len(body))),
        "log_primary_len": float(np.log1p(len(part["primary"]))),
        "log_ex99_len": float(np.log1p(len(part["ex99"]))),
        "amendment": float(str(row.get("form", "")).endswith("/A")),
        "signature_present": float(bool(SIGNATURE_RE.search(body))),
        "forward_disclaimer_present": float(bool(DISCLAIMER_RE.search(body))),
    }
    items = sorted(set(ITEM_RE.findall(body)))
    for item in items:
        result[f"item={item}"] = 1.0
    for left, right in zip(items, items[1:]):
        result[f"item_pair={left}|{right}"] = 1.0
    for name, pattern in TAXONOMY.items():
        regex = re.compile(pattern, re.I | re.S)
        total = len(regex.findall(text))
        result[f"tax={name}"] = float(min(total, 10))
        result[f"primary_tax={name}"] = float(bool(regex.search(part["primary"])))
        result[f"ex99_tax={name}"] = float(bool(regex.search(part["ex99"])))
    amounts = []
    for value, scale in AMOUNT_RE.findall(text):
        number = float(value) * {"million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}.get(scale.lower(), 1.0)
        if 0 < number < 1e13:
            amounts.append(number)
    result["max_amount_log"] = float(np.log1p(max(amounts))) if amounts else 0.0
    offer = OFFER_PRICE_RE.search(text)
    entry = float(row.get("log_entry_price", np.nan))
    if offer and np.isfinite(entry):
        result["offering_price_vs_entry"] = float(np.clip(float(offer.group(1)) / np.exp(entry) - 1, -2, 2))
    trial = [int(value) for value in TRIAL_N_RE.findall(text) if int(value) > 0]
    result["max_trial_n_log"] = float(np.log1p(max(trial))) if trial else 0.0
    pvalues = [float(value) for value in PVALUE_RE.findall(text) if float(value) > 0]
    result["min_pvalue_neglog"] = float(-np.log10(min(pvalues))) if pvalues else 0.0
    hazards = [float(value) for value in HR_RE.findall(text) if float(value) > 0]
    result["hazard_ratio_log"] = float(np.log(hazards[0])) if hazards else 0.0
    return result


def prepare_sec_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy().reset_index(drop=True)
    for column in ("headline", "body", "form", "event_type", "source_family"):
        if column not in prepared.columns:
            prepared[column] = ""
        prepared[column] = prepared[column].fillna("").astype(str)
    prepared["sec_segment_text"] = prepared.apply(lambda row: segments(row)["combined"], axis=1)
    prepared["sec_structure"] = prepared.apply(structured_features, axis=1)
    return prepared


def load_bundle(path: str | Path) -> dict[str, Any]:
    with gzip.open(Path(path), "rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("model_family") != MODEL_FAMILY:
        raise RuntimeError("V224_FROZEN_BUNDLE_IDENTITY_MISMATCH")
    return bundle


def raw_predict(frame: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    prepared = prepare_sec_features(frame)
    numeric = bundle["models"]["numeric_pipeline"].predict_proba(numeric_feature_frame(prepared))[:, 1]
    text_matrix = bundle["models"]["text_vectorizer"].transform(prepared["sec_segment_text"])
    text = bundle["models"]["text_classifier"].predict_proba(text_matrix)[:, 1]
    structure_matrix = bundle["models"]["semantic_structure_vectorizer"].transform(prepared["sec_structure"].tolist())
    structure = bundle["models"]["semantic_structure_classifier"].predict_proba(structure_matrix)[:, 1]
    parts = {"N": np.asarray(numeric, float), "T": np.asarray(text, float), "S": np.asarray(structure, float)}
    total = sum(BLEND_WEIGHTS.values())
    blended_logit = sum(
        weight * logit(np.clip(parts[name], PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP))
        for name, weight in BLEND_WEIGHTS.items()
    ) / total
    parts["raw_probability"] = expit(blended_logit)
    return parts


def select_cutoff(previous: dict[str, list[Any]]) -> float:
    oriented = np.asarray(previous["oriented_probability"], float)
    y = np.asarray(previous["y"], int)
    trials = []
    for cutoff in CUTOFF_GRID:
        accuracy = float(((oriented >= cutoff) == y).mean())
        trials.append((accuracy, -abs(float(cutoff) - 0.5), float(cutoff)))
    return max(trials)[2]


def confidence_score(previous: dict[str, list[Any]], low: bool) -> tuple[int, float]:
    margin = np.asarray(previous["margin"], float)
    threshold = float(np.quantile(margin, 1.0 - CONFIDENCE_QUANTILE if low else CONFIDENCE_QUANTILE))
    selected = margin <= threshold if low else margin >= threshold
    prediction = np.asarray(previous["prediction"], bool)[selected]
    y = np.asarray(previous["y"], int)[selected]
    returns = np.asarray(previous["fwd_ret_30m"], float)[selected]
    accuracy = float((prediction == y).mean())
    net = float((np.where(prediction, 1.0, -1.0) * returns - COST).mean())
    checks = int(accuracy >= 0.60) + int(net > 0)
    return checks, accuracy + 5.0 * float(np.clip(net, -0.01, 0.01))


def derive_policy(state: dict[str, Any]) -> dict[str, Any]:
    machine = state["machine_state"]
    raw = machine["cumulative_raw_history"]
    raw_y = np.asarray(raw["y"], int)
    raw_probability = np.asarray(raw["raw_probability"], float)
    if len(raw_y) and np.unique(raw_y).size == 2:
        prior_auc = float(roc_auc_score(raw_y, raw_probability))
        reverse = prior_auc < POLARITY_AUC_FLOOR
    else:
        prior_auc = None
        reverse = False
    previous = machine.get("previous_completed_block")
    cutoff = DIRECTION_THRESHOLD if previous is None else select_cutoff(previous)
    high_score = low_score = None
    use_low_margin = False
    if previous is not None:
        high_score = confidence_score(previous, low=False)
        low_score = confidence_score(previous, low=True)
        use_low_margin = low_score > high_score
    return {
        "prior_raw_rows": int(len(raw_y)),
        "prior_raw_auc": prior_auc,
        "polarity_auc_floor": POLARITY_AUC_FLOOR,
        "polarity_reversed": bool(reverse),
        "direction_cutoff": float(cutoff),
        "previous_high_margin_score": None if high_score is None else [int(high_score[0]), float(high_score[1])],
        "previous_low_margin_score": None if low_score is None else [int(low_score[0]), float(low_score[1])],
        "confidence_polarity": "LOW_MARGIN" if use_low_margin else "HIGH_MARGIN",
    }


def validate_score_input(frame: pd.DataFrame, require_block_size: bool = True) -> pd.DataFrame:
    forbidden = sorted(FORBIDDEN_SCORE_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"V224_LABEL_OR_OUTCOME_COLUMNS_FORBIDDEN_AT_SCORE_TIME:{','.join(forbidden)}")
    required = {"event_id", "event_time_utc", "source_family", "headline", "body", "form", "event_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V224_REQUIRED_SCORE_COLUMNS_MISSING:{','.join(missing)}")
    if require_block_size and len(frame) != CONFIRMATION_BLOCK_SIZE:
        raise ValueError(f"V224_CONFIRMATION_BLOCK_SIZE_MISMATCH:{len(frame)}")
    if not frame.source_family.fillna("").astype(str).eq("US_SEC").all():
        raise ValueError("V224_SOURCE_ROUTER_REJECTS_NON_US_SEC")
    if frame.event_id.astype(str).duplicated().any():
        raise ValueError("V224_DUPLICATE_EVENT_ID_IN_BLOCK")
    if "regular_session_eligible" in frame and not frame.regular_session_eligible.astype(bool).all():
        raise ValueError("V224_REGULAR_SESSION_ELIGIBILITY_FAIL")
    ordered = frame.copy()
    ordered["event_time_utc"] = pd.to_datetime(ordered.event_time_utc, utc=True, errors="raise")
    return ordered.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)


def state_fingerprint(state: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(state))


def score_block(frame: pd.DataFrame, bundle: dict[str, Any], state: dict[str, Any],
                require_block_size: bool = True) -> pd.DataFrame:
    """Score one label-blind chronological block without mutating state."""
    ordered = validate_score_input(frame, require_block_size=require_block_size)
    policy = derive_policy(state)
    parts = raw_predict(ordered, bundle)
    raw_probability = parts["raw_probability"]
    oriented = 1.0 - raw_probability if policy["polarity_reversed"] else raw_probability
    probability = np.clip(0.5 + oriented - policy["direction_cutoff"], 0.0, 1.0)
    prediction = probability >= DIRECTION_THRESHOLD
    margin = np.abs(probability - DIRECTION_THRESHOLD)
    use_low = policy["confidence_polarity"] == "LOW_MARGIN"
    threshold = float(np.quantile(
        margin, 1.0 - CONFIDENCE_QUANTILE if use_low else CONFIDENCE_QUANTILE,
    ))
    high_conf = margin <= threshold if use_low else margin >= threshold
    output = ordered[["event_id", "event_time_utc", "source_family", "form", "event_type"]].copy()
    output["numeric_probability"] = parts["N"]
    output["text_probability"] = parts["T"]
    output["semantic_structure_probability"] = parts["S"]
    output["raw_probability"] = raw_probability
    output["oriented_probability"] = oriented
    output["probability"] = probability
    output["prediction"] = prediction
    output["margin"] = margin
    output["high_conf"] = high_conf
    output["confidence_signal"] = -margin if use_low else margin
    output["polarity_reversed"] = policy["polarity_reversed"]
    output["direction_cutoff"] = policy["direction_cutoff"]
    output["confidence_polarity"] = policy["confidence_polarity"]
    output["confidence_threshold"] = threshold
    output["confirmation_block"] = int(state["machine_state"]["next_confirmation_block"])
    return output


def complete_block(state: dict[str, Any], scored: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    """Return the next state after a fully completed block; never mutates input state."""
    required = {"event_id", "y", "fwd_ret_30m"}
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"V224_OUTCOME_COLUMNS_MISSING:{','.join(missing)}")
    if scored.event_id.astype(str).duplicated().any() or outcomes.event_id.astype(str).duplicated().any():
        raise ValueError("V224_DUPLICATE_EVENT_ID_AT_COMPLETION")
    merged = scored.merge(outcomes[["event_id", "y", "fwd_ret_30m"]], on="event_id",
                          how="left", validate="one_to_one")
    if merged[["y", "fwd_ret_30m"]].isna().any().any() or len(merged) != len(outcomes):
        raise ValueError("V224_INCOMPLETE_OR_EXTRA_BLOCK_OUTCOMES")
    y = merged.y.astype(int)
    if not y.isin([0, 1]).all():
        raise ValueError("V224_INVALID_BINARY_OUTCOME")
    updated = copy.deepcopy(state)
    machine = updated["machine_state"]
    raw = machine["cumulative_raw_history"]
    raw["event_id"].extend(merged.event_id.astype(str).tolist())
    raw["raw_probability"].extend(merged.raw_probability.astype(float).tolist())
    raw["y"].extend(y.tolist())
    machine["previous_completed_block"] = {
        "event_id": merged.event_id.astype(str).tolist(),
        "oriented_probability": merged.oriented_probability.astype(float).tolist(),
        "guarded_probability": merged.probability.astype(float).tolist(),
        "prediction": merged.prediction.astype(bool).tolist(),
        "margin": merged.margin.astype(float).tolist(),
        "y": y.tolist(),
        "fwd_ret_30m": merged.fwd_ret_30m.astype(float).tolist(),
    }
    block = int(machine["next_confirmation_block"])
    machine["completed_confirmation_blocks"].append({"block": block, "n": int(len(merged))})
    machine["next_confirmation_block"] = block + 1
    machine["confirmation_rows_completed"] = int(machine["confirmation_rows_completed"]) + len(merged)
    return updated


def score_digest(frame: pd.DataFrame) -> str:
    columns = [
        "event_id", "numeric_probability", "text_probability", "semantic_structure_probability",
        "raw_probability", "oriented_probability", "probability", "prediction", "margin",
        "high_conf", "confidence_signal", "polarity_reversed", "direction_cutoff",
        "confidence_polarity", "confidence_threshold", "confirmation_block",
    ]
    records = []
    for row in frame[columns].itertuples(index=False, name=None):
        record = {}
        for name, value in zip(columns, row):
            if isinstance(value, (np.bool_, bool)):
                record[name] = bool(value)
            elif isinstance(value, (np.integer, int)):
                record[name] = int(value)
            elif isinstance(value, (np.floating, float)):
                record[name] = float(value)
            else:
                record[name] = str(value)
        records.append(record)
    return sha256_bytes(canonical_json_bytes(records))


def _read_events(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="infer", low_memory=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one frozen V224 label-blind US_SEC block")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    scored = score_block(_read_events(args.events), bundle, state)
    print(scored.to_json(orient="records", date_format="iso", double_precision=15), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
