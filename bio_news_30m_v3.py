#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIO_NEWS_30M_V3
================

목표
----
국장 + 해외장 바이오/제약/헬스케어 메이저 종목의 "뉴스/공시 공개 직후" 단기 반응을 예측한다.

PRIMARY TARGET
--------------
event timestamp 공개 -> +2분 latency 후 entry -> 그 시점부터 +30분 뒤 가격 방향
label = UP if close(t+32m) > close(t+2m), else DOWN

정규장 중 발생했고 +32분까지 정규장 데이터가 존재하는 이벤트만 PRIMARY SEAL에 사용.
장전/장후 이벤트는 NEXT_OPEN 버킷으로 별도 저장한다.

인증 원칙
---------
- 최소 150종목 universe
- 뉴스 timestamp가 분 단위 이상 정확한 이벤트만 사용
- DEV에서만 model/threshold 선택
- SEAL 데이터는 검색 코드로 전달하지 않음
- SEARCH 종료 후 code/data/model SHA256 동결
- SEAL 1회 오픈
- 결과를 보고 같은 seal에 맞춰 재튜닝 금지
- 단순 accuracy 외 balanced accuracy/AUC/high-confidence/naive edge/전략 signed return/bootstrap CI 검사
- US/KR 양쪽 최소 이벤트 수 충족 요구

데이터 소스 어댑터
------------------
US events:
  - SEC EDGAR submissions / 8-K, 6-K 등 exact acceptanceDateTime
  - optional generic exact-timestamp news CSV

KR events:
  - KIND exact disclosure timestamps via local CSV or krx-kind-data-api adapter
  - optional generic exact-timestamp news CSV

US 1m prices:
  - local HFDL parquet
  - optional HFDL_API_KEY auto-download

KR 1m prices:
  - local per-symbol parquet/csv
  - or one large KR parquet with symbol column
  - live recorder/증권사 API로 누적한 파일과 호환

이 프로그램은 실제 미래 수익을 보장하지 않는다.
PASS는 "미사용 봉인구간에서 사전 정의한 기준을 통과"했다는 뜻이다.
"""

from __future__ import annotations

import runtime_limits

import argparse
import hashlib
import io
import json
import math
import html as html_lib
import os
import re
import time
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP = "BIO_NEWS_30M_V35"
VERSION = "35.0.0"
SEED = 350826
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
UNIVERSE_FILE = ROOT / "universe_seed.csv"
DATA = ROOT / "data"
PRICE = ROOT / "price_data"
OUT = ROOT / "output" / "V35"
CERT = ROOT / "cert" / "V35"
CACHE = ROOT / "cache"
for p in (DATA, PRICE, OUT, CERT, CACHE):
    p.mkdir(parents=True, exist_ok=True)

EVENT_FILE = DATA / "events_exact_v35.csv.gz"
LABELED_FILE = DATA / "labeled_events_v35.csv.gz"
SOURCE_STATUS_FILE = DATA / "V35_SOURCE_STATUS.json"
FROZEN_SPEC = CERT / "FROZEN_SPEC.json"
FROZEN_MODEL = CERT / "FROZEN_MODEL.joblib"
SEALED_JSON = CERT / "SEALED_CERTIFICATE.json"
SEALED_TXT = CERT / "SEALED_CERTIFICATE.txt"

US_TZ = ZoneInfo("America/New_York")
KR_TZ = ZoneInfo("Asia/Seoul")

SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "BIO_NEWS_30M_V3 research contact@example.com"),
    "Accept-Encoding": "gzip, deflate",
}



@dataclass(frozen=True)
class Config:
    entry_lag_min: int = 2
    horizon_min: int = 30

    # V36 contract repair.  Older opened certificates retain their archived
    # source snapshots; callers must opt into this stricter path explicitly.
    official_exact_contract: bool = False
    price_contract_version: str = "EXACT_T2_30M_V36"
    max_official_cadence_sec: int = 60
    max_entry_slippage_sec: int = 60
    max_exit_slippage_sec: int = 60
    min_hold_seconds: int = 1800
    max_hold_seconds: int = 1860
    max_feature_staleness_sec: int = 60

    # event filtering
    min_universe: int = 150
    material_only: bool = True
    min_headline_chars: int = 3

    # split
    seal_fraction: float = 0.20
    # 210 days is fixed before labels/model search so the lower-frequency KR
    # disclosure stream clears the predeclared >=100 sealed-event requirement.
    seal_min_days: int = 210
    dev_min_days: int = 365
    cv_folds: int = 4
    # US and KR public intraday/news archives cover different calendar eras.
    # Each market therefore gets its own chronological untouched cutoff.
    market_stratified_split: bool = True
    # V35 is a label-blind pure external seal of exact-time events unused through V34.
    fixed_seal_sources: Tuple[Tuple[str, str], ...] = (
        ("US", "US_EXACT_V35_SEAL"),
        ("KR", "KR_EXACT_V35_SEAL"),
    )
    dev_proxy_sources: Tuple[Tuple[str, str], ...] = (
        ("US", "SEC"),
        ("US", "SEC_V7"),
        ("US", "SEC_V8"),
        ("US", "SEC_V9"),
        ("US", "SEC_V10"),
        ("US", "SEC_V11"),
        ("US", "SEC_V12_DEV"),
        ("US", "SEC_V12_SEAL"),
        ("US", "SEC_V13_DEV"),
        ("US", "SEC_V13_SEAL"),
        ("US", "SEC_V14_DEV"),
        ("US", "SEC_V14_SEAL"),
        ("US", "SEC_V15_DEV"),
        ("US", "SEC_V15_SEAL"),
        ("US", "SEC_V16_SEAL"),
        ("US", "SEC_V17_DEV"),
        ("US", "SEC_V17_SEAL"),
        ("US", "SEC_V18_DEV"),
        ("US", "SEC_V18_SEAL"),
        ("US", "SEC_V19_DEV"),
        ("US", "SEC_V19_SEAL"),
        ("US", "SEC_V20_DEV"),
        ("US", "SEC_V20_SEAL"),
        ("US", "SEC_V21_DEV"),
        ("US", "SEC_V21_SEAL"),
        ("US", "SEC_V22_DEV"),
        ("US", "SEC_V22_SEAL"),
        ("US", "SEC_V23_DEV"),
        ("US", "SEC_V23_SEAL"),
        ("US", "SEC_V24_DEV"),
        ("US", "SEC_V24_SEAL"),
        ("US", "SEC_V25_DEV"),
        ("US", "SEC_V25_SEAL"),
        ("US", "SEC_V26_DEV"),
        ("US", "SEC_V26_SEAL"),
        ("US", "SEC_V27_DEV"),
        ("US", "SEC_V27_SEAL"),
        ("US", "US_EXACT_V28_DEV"),
        ("US", "US_EXACT_V28_SEAL"),
        ("US", "US_EXACT_V29_DEV"),
        ("US", "US_EXACT_V29_SEAL"),
        ("US", "US_EXACT_V30_DEV"),
        ("US", "US_EXACT_V30_SEAL"),
        ("US", "US_EXACT_V31_DEV"),
        ("US", "US_EXACT_V31_SEAL"),
        ("US", "US_EXACT_V32_DEV"),
        ("US", "US_EXACT_V32_SEAL"),
        ("US", "US_EXACT_V33_DEV"),
        ("US", "US_EXACT_V33_SEAL"),
        ("US", "US_EXACT_V34_SEAL"),
        ("KR", "NAVER_NEWS"),
        ("KR", "NAVER_NEWS_V7"),
        ("KR", "NAVER_NEWS_V8"),
        ("KR", "NAVER_NEWS_V9_RESERVED"),
        ("KR", "NAVER_NEWS_V10"),
        ("KR", "NAVER_NEWS_V11_RESERVED"),
        ("KR", "NAVER_NEWS_V12_DEV"),
        ("KR", "NAVER_NEWS_V12_SEAL"),
        ("KR", "NAVER_NEWS_V13_DEV"),
        ("KR", "NAVER_NEWS_V13_SEAL"),
        ("KR", "NAVER_NEWS_V14_DEV"),
        ("KR", "NAVER_NEWS_V14_SEAL"),
        ("KR", "KIND_V15_DEV"),
        ("KR", "KIND_V15_SEAL"),
        ("KR", "KIND_V16_SEAL"),
        ("KR", "KIND_V17_DEV"),
        ("KR", "KIND_V17_SEAL"),
        ("KR", "KIND_V18_DEV"),
        ("KR", "KIND_V18_SEAL"),
        ("KR", "KIND_V19_DEV"),
        ("KR", "KIND_V19_SEAL"),
        ("KR", "NAVER_NEWS_V20_DEV"),
        ("KR", "NAVER_NEWS_V20_SEAL"),
        ("KR", "NAVER_NEWS_V21_DEV"),
        ("KR", "NAVER_NEWS_V21_SEAL"),
        ("KR", "NAVER_NEWS_V22_DEV"),
        ("KR", "NAVER_NEWS_V22_SEAL"),
        ("KR", "NAVER_NEWS_V23_DEV"),
        ("KR", "NAVER_NEWS_V23_SEAL"),
        ("KR", "NAVER_NEWS_V24_DEV"),
        ("KR", "NAVER_NEWS_V24_SEAL"),
        ("KR", "NAVER_NEWS_V25_DEV"),
        ("KR", "NAVER_NEWS_V25_SEAL"),
        ("KR", "NAVER_NEWS_V26_DEV"),
        ("KR", "NAVER_NEWS_V26_SEAL"),
        ("KR", "NAVER_NEWS_V27_DEV"),
        ("KR", "NAVER_NEWS_V27_SEAL"),
        ("KR", "KR_EXACT_V28_DEV"),
        ("KR", "KR_EXACT_V28_SEAL"),
        ("KR", "KR_EXACT_V29_DEV"),
        ("KR", "KR_EXACT_V29_SEAL"),
        ("KR", "KR_EXACT_V30_DEV"),
        ("KR", "KR_EXACT_V30_SEAL"),
        ("KR", "KR_EXACT_V31_DEV"),
        ("KR", "KR_EXACT_V31_SEAL"),
        ("KR", "KR_EXACT_V32_DEV"),
        ("KR", "KR_EXACT_V32_SEAL"),
        ("KR", "KR_EXACT_V33_DEV"),
        ("KR", "KR_EXACT_V33_SEAL"),
        ("KR", "KR_EXACT_V34_SEAL"),
    )
    # V35 is a pure external seal. V34 is now opened for development only.
    dev_validation_sources: Tuple[Tuple[str, str], ...] = (
        ("KR", "KR_EXACT_V34_SEAL"),
        ("US", "US_EXACT_V34_SEAL"),
    )
    final_training_sources: Tuple[Tuple[str, str], ...] = (
        ("KR", "KR_EXACT_V34_SEAL"),
        ("US", "US_EXACT_V34_SEAL"),
    )
    dev_market_weight_targets: Tuple[Tuple[str, float], ...] = (
        ("KR", 171.0),
        ("US", 617.0),
    )
    decision_cutoff_by_market: Tuple[Tuple[str, float], ...] = (
        ("US", 0.500),
        ("KR", 0.500),
    )
    invert_probability_by_market: Tuple[Tuple[str, bool], ...] = (
        ("US", False),
        ("KR", False),
    )
    fixed_seal_windows: Tuple[Tuple[str, str, str], ...] = ()

    fixed_highconf_threshold: Optional[float] = 0.925

    # transaction interpretation only
    round_trip_cost: float = 0.0020

    # cert requirements
    cert_min_total_events: int = 500
    cert_min_us_events: int = 250
    cert_min_kr_events: int = 100
    cert_min_bal_acc: float = 0.540
    cert_min_auc: float = 0.560
    cert_min_edge_vs_naive: float = 0.020
    cert_min_highconf_coverage: float = 0.15
    cert_min_highconf_acc: float = 0.600
    cert_min_mean_signed_net: float = 0.0000
    cert_min_boot_bal_lower95: float = 0.500
    cert_min_market_bal_acc: float = 0.510

    # source paths
    us_price_dir: str = "price_data/US"
    kr_price_dir: str = "price_data/KR"
    kr_big_parquet: str = "price_data/KR_ALL_1M.parquet"

    # SEC
    sec_start: str = "2019-01-01"
    sec_forms: Tuple[str, ...] = (
        "8-K", "8-K/A", "6-K", "6-K/A",
        "10-Q", "10-Q/A", "10-K", "10-K/A",
        "S-3", "S-3ASR", "424B3", "424B5",
        "20-F", "20-F/A", "F-1", "F-1/A", "S-1", "S-1/A",
        "424B4", "DEF 14A", "DEFA14A", "SC 13D", "SC 13D/A",
    )


NUMERIC_COLS = [
    "pre_ret_1", "pre_ret_2", "pre_ret_3", "pre_ret_5", "pre_ret_10",
    "pre_ret_15", "pre_ret_20", "pre_ret_30", "pre_ret_45", "pre_ret_60",
    "pre_ret_90", "pre_ret_120",
    "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
    "volume_ratio_1_30", "volume_ratio_2_30", "volume_ratio_5_30", "volume_ratio_10_60",
    "range_5m", "range_10m", "range_30m", "range_60m",
    "close_position_10", "close_position_30", "close_position_60",
    "entry_bar_ret", "entry_bar_range", "intraday_ret_open",
    "return_autocorr_30", "up_fraction_10", "up_fraction_30",
    "trend_slope_30", "trend_slope_60", "vwap_distance_30", "log_entry_price",
    "minutes_from_open", "minutes_to_close",
    "event_positive_kw", "event_negative_kw", "event_financing_kw",
    "event_trial_kw", "event_regulatory_kw", "event_ma_kw", "event_earnings_kw",
    "headline_len", "body_len",
    "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15", "benchmark_ret_30", "benchmark_ret_60",
    "excess_ret_5", "excess_ret_15", "excess_ret_30", "excess_ret_60",
    "event_hour", "event_weekday", "event_month", "event_hour_sin", "event_hour_cos",
    "abs_pre_ret_2", "abs_pre_ret_5", "abs_pre_ret_15",
    "abs_pre_ret_30", "abs_pre_ret_60",
    "ret_slope_2_15", "ret_slope_5_30", "kw_sentiment",
]
CAT_COLS = ["market", "source", "event_type", "ticker"]
TEXT_COL = "text"

def model_frame(frame:pd.DataFrame)->pd.DataFrame:
    """Create label-independent V7 model features from an event frame."""
    x=frame.copy()
    for column in (TEXT_COL,*NUMERIC_COLS,*CAT_COLS):
        if column not in x.columns:
            x[column]=np.nan if column in NUMERIC_COLS else ""
    ts=pd.to_datetime(x["event_time_utc"],utc=True)
    x["event_hour"]=ts.dt.hour+ts.dt.minute/60.0
    x["event_weekday"]=ts.dt.weekday
    x["event_month"]=ts.dt.month
    x["event_hour_sin"]=np.sin(2.0*np.pi*x["event_hour"]/24.0)
    x["event_hour_cos"]=np.cos(2.0*np.pi*x["event_hour"]/24.0)
    for suffix in ("2","5","15","30","60"):
        x[f"abs_pre_ret_{suffix}"]=pd.to_numeric(
            x[f"pre_ret_{suffix}"],errors="coerce"
        ).abs()
    x["ret_slope_2_15"]=x["pre_ret_2"]-x["pre_ret_15"]
    x["ret_slope_5_30"]=x["pre_ret_5"]-x["pre_ret_30"]
    x["kw_sentiment"]=(
        x["event_positive_kw"]-x["event_negative_kw"]-x["event_financing_kw"]
    )
    for suffix in ("5","15","30","60"):
        x[f"excess_ret_{suffix}"]=(
            pd.to_numeric(x[f"pre_ret_{suffix}"],errors="coerce")-
            pd.to_numeric(x[f"benchmark_ret_{suffix}"],errors="coerce")
        )
    passthrough=[column for column in ("event_id","event_time_utc") if column in x.columns]
    return x[[TEXT_COL]+NUMERIC_COLS+CAT_COLS+passthrough]

def shift_probability(probability:np.ndarray,cutoff:float)->np.ndarray:
    p=np.clip(np.asarray(probability,dtype=float),1e-6,1-1e-6)
    logit=np.log(p/(1-p))-math.log(cutoff/(1-cutoff))
    return 1.0/(1.0+np.exp(-logit))

POSITIVE_KW = [
    "approval","approved","positive","met primary","met its primary","statistically significant",
    "breakthrough","fast track","priority review","orphan drug","successful","success",
    "license agreement","partnership","collaboration","milestone payment","acquisition","merger",
    "허가","승인","긍정","유효성","통계적 유의","기술수출","라이선스","계약 체결","임상 성공","목표 달성",
]
NEGATIVE_KW = [
    "complete response letter","crl","clinical hold","failed","failure","missed primary",
    "did not meet","adverse event","serious adverse","terminated","discontinued","safety concern",
    "warning letter","recall","rejected","denied","downgrade",
    "임상 중단","임상 실패","목표 미달","안전성","중대한 이상반응","반려","거절","회수","경고",
]
FINANCING_KW = [
    "public offering","registered direct","private placement","at-the-market","atm offering",
    "convertible","warrant","dilution","shelf registration","424b5","s-3",
    "유상증자","전환사채","신주인수권","제3자배정","증권신고서","주식 발행","희석",
]
TRIAL_KW = [
    "phase 1","phase i","phase 2","phase ii","phase 3","phase iii","clinical trial",
    "topline","top-line","primary endpoint","secondary endpoint","patient enrollment",
    "임상 1상","임상1상","임상 2상","임상2상","임상 3상","임상3상","임상시험","탑라인","유효성",
]
REGULATORY_KW = [
    "fda","ema","nda","bla","ind","pdufa","approval","crl","clinical hold",
    "식약처","품목허가","허가 신청","허가 승인","임상시험계획","ind 승인",
]
MA_KW = [
    "merger","acquisition","acquire","tender offer","strategic alternatives",
    "합병","인수","피인수","공개매수","경영권",
]
EARNINGS_KW = [
    "earnings","revenue","guidance","quarter results","financial results","eps",
    "실적","매출","영업이익","당기순이익","잠정실적","가이던스",
]

# The inherited source had replacement characters in its Korean literals.
# Append valid UTF-8 terms so Korean material-event filtering and keyword
# features operate on the actual article text.
POSITIVE_KW += [
    "승인", "허가", "긍정", "유효성", "통계적 유의", "기술수출", "기술이전",
    "라이선스", "계약 체결", "임상 성공", "목표 달성", "품목허가",
    "우선심사", "혁신신약", "연구 결과", "특허", "독점",
]
NEGATIVE_KW += [
    "임상 중단", "임상 실패", "목표 미달", "안전성", "중대한 이상반응",
    "반려", "거절", "회수", "경고", "판매중지", "허가 취소",
]
FINANCING_KW += [
    "유상증자", "전환사채", "신주인수권", "제3자배정", "증권신고서",
    "주식 발행", "자금조달",
]
TRIAL_KW += [
    "임상 1상", "임상1상", "임상 2상", "임상2상", "임상 3상", "임상3상",
    "임상시험", "톱라인", "주요평가변수", "환자 등록",
]
REGULATORY_KW += [
    "식약처", "품목허가", "허가 신청", "허가 승인", "임상시험계획", "IND 승인",
]
MA_KW += ["합병", "인수", "피인수", "공개매수", "경영권"]
EARNINGS_KW += [
    "실적", "매출", "영업이익", "분기 실적", "잠정실적", "가이던스",
]

MATERIAL_PATTERNS = POSITIVE_KW + NEGATIVE_KW + FINANCING_KW + TRIAL_KW + REGULATORY_KW + MA_KW + EARNINGS_KW

# V12's narrow Korean business/biopharma source definition was frozen from
# headline/body metadata before prices or labels were joined.
V12_KR_BUSINESS_KW = [
    "\uacf5\uae09","\uc218\uc8fc","\uc5c5\ubb34\ud611\uc57d","\ud611\uc57d","MOU","\ucd9c\uc2dc",
    "\uc2e0\uc57d","\uce58\ub8cc\uc81c","\ud6c4\ubcf4\ubb3c\uc9c8","\ub17c\ubb38","\ud559\ud68c","\uc18c\uc1a1",
    "\uc0dd\uc0b0","\uacf5\uc7a5","\uc81c\ud488","\uc9c4\ub2e8","\ubc31\uc2e0","\ud751\uc790","\uc801\uc790",
    "\uacc4\uc57d","\ud30c\ud2b8\ub108","\ubc14\uc774\uc624\uc2dc\ubc00\ub7ec","\uac74\uac15\ubcf4\ud5d8","\ud574\uc678","\uc784\uc0c1",
]
MATERIAL_PATTERNS += V12_KR_BUSINESS_KW
V13_KR_BUSINESS_KW = [
    "\ud22c\uc790","\uc0c1\uc7a5","\uc5f0\uad6c","\uac1c\ubc1c","\uae30\uc220","\uae00\ub85c\ubc8c",
    "\ud2b9\ud5c8","\ub3c5\uc810","FDA","\uc2b9\uc778","\ud5c8\uac00","\ub9e4\ucd9c","\uc2e4\uc801",
    "\uc601\uc5c5\uc774\uc775","\ud569\ubcd1","\uc778\uc218","\uc99d\uc790","\uc804\ud658\uc0ac\ucc44",
]
MATERIAL_PATTERNS += V13_KR_BUSINESS_KW




def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()

def code_sha() -> str:
    return sha256_file(Path(__file__).resolve())

def norm_ticker(x: str, market: str) -> str:
    x = str(x).strip().upper()
    if market == "KR":
        return re.sub(r"\D", "", x).zfill(6)
    return x.replace(".", "-")

def safe_auc(y, p, sample_weight=None) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, sample_weight=sample_weight))

def count_kw(text: str, words: Sequence[str]) -> int:
    t = (text or "").lower()
    return sum(1 for w in words if w.lower() in t)

def infer_event_type(text: str, form: str = "") -> str:
    t = (text or "").lower()
    form = "" if pd.isna(form) else str(form)
    scores = {
        "REGULATORY": count_kw(t, REGULATORY_KW),
        "CLINICAL": count_kw(t, TRIAL_KW),
        "FINANCING": count_kw(t, FINANCING_KW),
        "M&A": count_kw(t, MA_KW),
        "EARNINGS": count_kw(t, EARNINGS_KW),
        "NEGATIVE_SAFETY": count_kw(t, NEGATIVE_KW),
    }
    if form.upper() in {"S-3","S-3ASR","424B3","424B5"}:
        scores["FINANCING"] += 3
    if form.upper().startswith("10-Q") or form.upper().startswith("10-K"):
        scores["EARNINGS"] += 1
    best, score = max(scores.items(), key=lambda kv: kv[1])
    return best if score > 0 else ("SEC_" + form.upper() if form else "OTHER")

def is_material(text: str, form: str = "") -> bool:
    form = "" if pd.isna(form) else str(form)
    if form.upper() in {"S-3","S-3ASR","424B3","424B5"}:
        return True
    return count_kw(text, MATERIAL_PATTERNS) > 0

def parse_utc(s: Any) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def session_bounds(event_utc: pd.Timestamp, market: str) -> Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp]:
    tz = US_TZ if market == "US" else KR_TZ
    local = event_utc.tz_convert(tz)
    d = local.date()
    if market == "US":
        op = pd.Timestamp(datetime(d.year,d.month,d.day,9,30,tzinfo=tz))
        cl = pd.Timestamp(datetime(d.year,d.month,d.day,16,0,tzinfo=tz))
    else:
        op = pd.Timestamp(datetime(d.year,d.month,d.day,9,0,tzinfo=tz))
        cl = pd.Timestamp(datetime(d.year,d.month,d.day,15,30,tzinfo=tz))
    return local, op, cl




def load_seed_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_FILE, dtype={"ticker":str})
    df["market"] = df["market"].str.upper()
    df["ticker"] = [norm_ticker(t,m) for t,m in zip(df["ticker"],df["market"])]
    return df.drop_duplicates(["market","ticker"]).reset_index(drop=True)

def build_live_universe(min_n: int = 150) -> pd.DataFrame:
    """
    FinanceDataReader listing metadata가 있으면 현재 US/KR bio/health universe를 확장한다.
    실패 시 seed를 그대로 사용.
    """
    seed = load_seed_universe()
    frames = [seed]
    try:
        import FinanceDataReader as fdr

        # KR
        for listing_name in ("KRX",):
            try:
                k = fdr.StockListing(listing_name)
                if k is not None and not k.empty:
                    cols = {str(c).lower(): c for c in k.columns}
                    symc = cols.get("code") or cols.get("symbol")
                    namec = cols.get("name")
                    sectorc = cols.get("sector")
                    industryc = cols.get("industry")
                    marcapc = cols.get("marcap") or cols.get("marketcap")
                    if symc is not None:
                        txt = pd.Series("", index=k.index, dtype=str)
                        for c in (namec, sectorc, industryc):
                            if c is not None:
                                txt = txt + " " + k[c].fillna("").astype(str)
                        mask = txt.str.contains(
                            r"바이오|제약|의약|생명과학|헬스케어|진단|의료|유전자|세포|백신|신약",
                            case=False, regex=True
                        )
                        z = k[mask].copy()
                        if marcapc is not None:
                            z["_mc"] = pd.to_numeric(z[marcapc], errors="coerce")
                            z = z.sort_values("_mc", ascending=False).head(120)
                        add = pd.DataFrame({
                            "market":"KR",
                            "ticker":z[symc].astype(str).str.zfill(6),
                            "company":z[namec].astype(str) if namec is not None else "",
                            "priority":"dynamic",
                        })
                        frames.append(add)
            except Exception:
                pass

        # US listings
        for listing_name in ("NASDAQ","NYSE","AMEX"):
            try:
                s = fdr.StockListing(listing_name)
                if s is None or s.empty:
                    continue
                cols = {str(c).lower(): c for c in s.columns}
                symc = cols.get("symbol")
                namec = cols.get("name")
                industryc = cols.get("industry")
                sectorc = cols.get("sector")
                marcapc = cols.get("marketcap")
                if symc is None:
                    continue
                txt = pd.Series("", index=s.index, dtype=str)
                for c in (namec, industryc, sectorc):
                    if c is not None:
                        txt = txt + " " + s[c].fillna("").astype(str)
                mask = txt.str.contains(
                    r"biotech|biotechnology|pharma|pharmaceutical|drug|therapeutic|life science|genomic|health care|healthcare",
                    case=False, regex=True
                )
                z = s[mask].copy()
                if marcapc is not None:
                    z["_mc"] = pd.to_numeric(z[marcapc], errors="coerce")
                    z = z.sort_values("_mc", ascending=False).head(180)
                add = pd.DataFrame({
                    "market":"US",
                    "ticker":z[symc].astype(str).str.upper(),
                    "company":z[namec].astype(str) if namec is not None else "",
                    "priority":"dynamic",
                })
                frames.append(add)
            except Exception:
                pass
    except Exception:
        pass

    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = [norm_ticker(t,m) for t,m in zip(out["ticker"],out["market"])]
    out = out.drop_duplicates(["market","ticker"], keep="last")
    if len(out) < min_n:
        raise RuntimeError(f"bio/health universe={len(out)} < required {min_n}")
    out.to_csv(DATA / "universe_live.csv", index=False, encoding="utf-8-sig")
    return out




class SECCollector:
    TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
    ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(SEC_HEADERS)
        self._last = 0.0
        self.text_cache = CACHE / "sec_text"
        self.text_cache.mkdir(parents=True, exist_ok=True)

    def throttle(self):
        # SEC fair-access를 보수적으로 지키기 위해 초당 약 6~7 request 이하.
        gap = time.time() - self._last
        if gap < 0.16:
            time.sleep(0.16-gap)
        self._last = time.time()

    def get_json(self, url: str) -> dict:
        self.throttle()
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_text(self, url: str) -> str:
        self.throttle()
        r = self.session.get(url, timeout=35)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text

    @staticmethod
    def html_to_text(raw: str, max_chars: int = 35000) -> str:
        if not raw:
            return ""
        s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
        s = re.sub(r"(?is)<br\s*/?>", "\n", s)
        s = re.sub(r"(?is)</p\s*>", "\n", s)
        s = re.sub(r"(?is)<[^>]+>", " ", s)
        s = html_lib.unescape(s)
        s = re.sub(r"[ \t\r\f\v]+", " ", s)
        s = re.sub(r"\n\s*\n+", "\n", s)
        return s.strip()[:max_chars]

    def ticker_to_cik(self) -> Dict[str,int]:
        obj = self.get_json(self.TICKER_MAP)
        out={}
        for _,r in obj.items():
            out[str(r["ticker"]).upper().replace(".","-")] = int(r["cik_str"])
        return out

    def filing_text(self, cik: int, accession: str, primary: str, form: str) -> str:
        """
        SEC filing 본문 + 8-K/6-K의 EX-99 계열 보도자료를 합친다.
        accession별 로컬 cache를 사용하므로 재실행 때 SEC를 반복 호출하지 않는다.
        """
        safe_acc = str(accession).replace("-", "")
        cp = self.text_cache / f"{cik}_{safe_acc}.txt"
        if cp.exists():
            try:
                return cp.read_text(encoding="utf-8")[:60000]
            except Exception:
                pass

        base = self.ARCHIVES.format(cik=cik, acc_nodash=safe_acc)
        chunks = []

        # primary filing
        if primary:
            try:
                raw = self.get_text(f"{base}/{primary}")
                chunks.append(self.html_to_text(raw, 35000))
            except Exception:
                pass

        # 8-K / 6-K에서 실제 보도자료가 EX-99.1 등에 붙는 경우가 많다.
        if form.upper().startswith(("8-K", "6-K")):
            try:
                idx = self.get_json(f"{base}/index.json")
                items = idx.get("directory", {}).get("item", []) or []
                candidates = []
                for it in items:
                    name = str(it.get("name", ""))
                    lname = name.lower()
                    if not re.search(r"\.(htm|html|txt)$", lname):
                        continue
                    score = 0
                    if re.search(r"ex(?:hibit)?[-_ ]?99", lname):
                        score += 10
                    if "99.1" in lname or "99_1" in lname or "ex991" in lname:
                        score += 8
                    if any(k in lname for k in ("press", "release", "news")):
                        score += 5
                    if name == primary:
                        score -= 10
                    if score > 0:
                        candidates.append((score, name))
                candidates.sort(reverse=True)

                for _, name in candidates[:2]:
                    try:
                        raw = self.get_text(f"{base}/{name}")
                        chunks.append(self.html_to_text(raw, 25000))
                    except Exception:
                        pass
            except Exception:
                pass

        merged = "\n".join(x for x in chunks if x).strip()[:60000]
        try:
            cp.write_text(merged, encoding="utf-8")
        except Exception:
            pass
        return merged

    def collect(self, universe: pd.DataFrame) -> pd.DataFrame:
        us = universe[universe["market"]=="US"].copy()
        cmap = self.ticker_to_cik()
        cutoff = pd.Timestamp(self.cfg.sec_start, tz="UTC")
        rows=[]

        for i, row in us.iterrows():
            ticker=row["ticker"]
            cik=cmap.get(ticker)
            if cik is None:
                continue
            try:
                js=self.get_json(self.SUBMISSIONS.format(cik=cik))
                recent=js.get("filings",{}).get("recent",{})
                if not recent:
                    continue
                n=len(recent.get("accessionNumber",[]))
                for j in range(n):
                    form=str(recent.get("form",[""]*n)[j])
                    if form not in self.cfg.sec_forms:
                        continue

                    # 30분 event study는 분 단위 시각이 없는 filingDate를 쓰지 않는다.
                    arr = recent.get("acceptanceDateTime", [])
                    acc_dt = arr[j] if j < len(arr) else ""
                    if not acc_dt or "T" not in str(acc_dt):
                        continue

                    ts=pd.Timestamp(str(acc_dt))
                    if ts.tzinfo is None:
                        # SEC acceptanceDateTime은 일반적으로 Eastern time 표기로 내려오므로
                        # timezone 표기가 없을 때만 ET로 해석한다.
                        ts=ts.tz_localize("America/New_York").tz_convert("UTC")
                    else:
                        ts=ts.tz_convert("UTC")
                    if ts < cutoff:
                        continue

                    items=""
                    arr=recent.get("items",[])
                    if j < len(arr):
                        items=str(arr[j] or "")
                    primary=""
                    arr=recent.get("primaryDocument",[])
                    if j < len(arr):
                        primary=str(arr[j] or "")
                    acc=recent.get("accessionNumber",[""]*n)[j]

                    body = self.filing_text(cik, acc, primary, form)
                    headline=f"{form} {items}".strip()
                    text=f"{headline} {body}".strip()

                    # material_only가 켜진 상태라면 제출문서 텍스트까지 읽은 뒤
                    # 실제 바이오 이벤트/희석/실적 성격이 있는 경우만 남긴다.
                    if self.cfg.material_only and not is_material(text, form):
                        continue

                    rows.append({
                        "event_id":f"SEC:{cik}:{acc}",
                        "market":"US","ticker":ticker,
                        "company":str(row.get("company","") or js.get("name","")),
                        "event_time_utc":ts.isoformat(),
                        "source":"SEC",
                        "form":form,
                        "headline":headline,
                        "body":body,
                        "event_type":infer_event_type(text,form),
                        "timestamp_quality":"EXACT",
                        "url":f"https://www.sec.gov/Archives/edgar/data/{cik}/{str(acc).replace('-','')}/",
                    })
            except Exception as e:
                print(f"[SEC WARN] {ticker}: {e}")

        return pd.DataFrame(rows)




def normalize_kind_csv(path: Path) -> pd.DataFrame:
    """
    사용자가 krx-kind-data-api 또는 다른 수집기로 만든 CSV를 표준 schema로 변환.
    허용 column aliases를 최대한 자동 인식.
    """
    x = pd.read_csv(path, dtype=str)
    low={str(c).lower():c for c in x.columns}

    def pick(*names):
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None

    tc=pick("ticker","code","종목코드","종목")
    timec=pick("event_time","event_time_utc","datetime","공시시간","시간","date_time")
    titlec=pick("headline","title","공시제목","제목","report_nm")
    companyc=pick("company","회사명","corp_name","법인명")
    typec=pick("event_type","공시유형","type","유형")
    urlc=pick("url","공시url","link")
    if tc is None or timec is None or titlec is None:
        raise ValueError("KIND CSV에는 ticker/공시시간/공시제목에 해당하는 열이 필요합니다.")

    rows=[]
    for i,r in x.iterrows():
        ticker=norm_ticker(r[tc],"KR")
        raw=str(r[timec]).strip()
        ts=pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts=ts.tz_localize(KR_TZ)
        ts=ts.tz_convert("UTC")
        title=str(r[titlec] or "")
        evtype=str(r[typec]) if typec else infer_event_type(title)
        rows.append({
            "event_id":f"KIND:{ticker}:{ts.isoformat()}:{i}",
            "market":"KR","ticker":ticker,
            "company":str(r[companyc]) if companyc else "",
            "event_time_utc":ts.isoformat(),
            "source":"KIND",
            "form":"",
            "headline":title,
            "body":"",
            "event_type":evtype or infer_event_type(title),
            "timestamp_quality":"EXACT",
            "url":str(r[urlc]) if urlc else "",
        })
    return pd.DataFrame(rows)

def normalize_generic_news_csv(path: Path) -> pd.DataFrame:
    """
    columns:
      market,ticker,event_time_utc(or event_time),headline
    optional:
      company,body,source,event_type,url
    """
    x=pd.read_csv(path,dtype=str)
    required={"market","ticker","headline"}
    if not required.issubset(set(x.columns)):
        raise ValueError(f"generic news CSV missing {required-set(x.columns)}")
    tc="event_time_utc" if "event_time_utc" in x.columns else "event_time"
    if tc not in x.columns:
        raise ValueError("event_time_utc 또는 event_time 필요")
    rows=[]
    for i,r in x.iterrows():
        market=str(r["market"]).upper()
        ticker=norm_ticker(r["ticker"],market)
        ts=pd.Timestamp(r[tc])
        if ts.tzinfo is None:
            tz=US_TZ if market=="US" else KR_TZ
            ts=ts.tz_localize(tz)
        ts=ts.tz_convert("UTC")
        headline=str(r["headline"])
        body=str(r.get("body","") or "")
        source=str(r.get("source","GENERIC") or "GENERIC")
        evtype=str(r.get("event_type","") or infer_event_type(headline+" "+body))
        rows.append({
            "event_id":f"GEN:{market}:{ticker}:{ts.isoformat()}:{i}",
            "market":market,"ticker":ticker,
            "company":str(r.get("company","") or ""),
            "event_time_utc":ts.isoformat(),
            "source":source,"form":"",
            "headline":headline,"body":body,
            "event_type":evtype,
            "timestamp_quality":"EXACT",
            "url":str(r.get("url","") or ""),
        })
    return pd.DataFrame(rows)

def combine_events(frames: Sequence[pd.DataFrame], cfg: Config) -> pd.DataFrame:
    frames=[x for x in frames if x is not None and not x.empty]
    if not frames:
        raise RuntimeError("event frames empty")
    x=pd.concat(frames,ignore_index=True)
    x["market"]=x["market"].str.upper()
    x["ticker"]=[norm_ticker(t,m) for t,m in zip(x["ticker"],x["market"])]
    # Sources use both ``T``-separated ISO strings and space-separated strings
    # with offsets.  Pandas 2's single-format inference would otherwise turn
    # every row from the later format into NaT after concatenation.
    x["event_time_utc"]=pd.to_datetime(
        x["event_time_utc"],utc=True,errors="coerce",format="mixed"
    )
    x=x.dropna(subset=["event_time_utc","ticker","headline"])
    x=x[x["timestamp_quality"].eq("EXACT")]
    x=x[x["headline"].fillna("").str.len()>=cfg.min_headline_chars]
    x["text"]=(x["headline"].fillna("")+" "+x["body"].fillna("")).str.strip()
    if cfg.material_only:
        mask=[
            is_material(t,f)
            for t,f in zip(x["text"],x.get("form",pd.Series("",index=x.index)))
        ]
        x=x[np.asarray(mask)].copy()

    # duplicate story/ticker in 10 minute window: first public appearance only
    x=x.sort_values(["ticker","event_time_utc"])
    x["headline_norm"]=x["headline"].str.lower().str.replace(r"\W+"," ",regex=True).str.strip()
    keep=np.ones(len(x),dtype=bool)
    last={}
    for k,(_,r) in enumerate(x.iterrows()):
        key=(r["market"],r["ticker"],r["headline_norm"][:120])
        prev=last.get(key)
        if prev is not None and (r["event_time_utc"]-prev).total_seconds()<600:
            keep[k]=False
        else:
            last[key]=r["event_time_utc"]
    x=x.iloc[np.where(keep)[0]].copy()
    x=x.drop(columns=["headline_norm"],errors="ignore")
    x.to_csv(EVENT_FILE,index=False,compression="gzip",encoding="utf-8")
    return x




def normalize_bar_df(df: pd.DataFrame, market: str, ticker: str) -> pd.DataFrame:
    x=df.copy()
    low={str(c).lower():c for c in x.columns}
    dtc=None
    for n in ("datetime","timestamp","time","date"):
        if n in low:
            dtc=low[n];break
    if dtc is None:
        if isinstance(x.index,pd.DatetimeIndex):
            x=x.reset_index().rename(columns={x.index.name or "index":"datetime"})
            dtc="datetime"
        else:
            raise ValueError(f"{ticker}: no datetime column")
    closec=low.get("close") or low.get("price")
    openc=low.get("open")
    highc=low.get("high")
    lowc=low.get("low")
    volc=low.get("volume")
    if closec is None:
        raise ValueError(f"{ticker}: close missing")
    out=pd.DataFrame({
        "datetime":pd.to_datetime(x[dtc],errors="coerce"),
        "open":pd.to_numeric(x[openc],errors="coerce") if openc else np.nan,
        "high":pd.to_numeric(x[highc],errors="coerce") if highc else np.nan,
        "low":pd.to_numeric(x[lowc],errors="coerce") if lowc else np.nan,
        "close":pd.to_numeric(x[closec],errors="coerce"),
        "volume":pd.to_numeric(x[volc],errors="coerce") if volc else 0.0,
    }).dropna(subset=["datetime","close"])

    if out["datetime"].dt.tz is None:
        # HFDL is commonly timestamped in US market time/UTC depending file version.
        # Environment override allows exact specification.
        env_name = "US_PRICE_TZ" if market=="US" else "KR_PRICE_TZ"
        tzname=os.getenv(env_name, "America/New_York" if market=="US" else "Asia/Seoul")
        out["datetime"]=out["datetime"].dt.tz_localize(tzname,ambiguous="NaT",nonexistent="shift_forward").dt.tz_convert("UTC")
    else:
        out["datetime"]=out["datetime"].dt.tz_convert("UTC")
    out=out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    return out.reset_index(drop=True)


def price_contract_metadata(path:Path,market:str)->dict[str,Any]:
    """Return the frozen, label-independent semantics for a local bar store.

    A directory name is part of the data contract.  We never infer a 1-minute
    source merely because two adjacent observations happen to be one minute
    apart: sparse trading and missing bars would make that inference unsafe.
    """
    normalized=path.parent.name.upper()
    if market=="US" and normalized=="US":
        return {
            "price_source":"GGADDAM_FINNHUB_OHLCV_1M",
            "price_cadence_sec":60,
            "bar_timestamp_semantics":"OPEN",
            "official_exact_eligible":True,
        }
    if market=="KR" and normalized=="KR_V36_1M":
        return {
            "price_source":"YAHOO_CHART_KR_1M",
            "price_cadence_sec":60,
            "bar_timestamp_semantics":"OPEN",
            "official_exact_eligible":True,
        }
    if market=="US" and normalized.startswith("US_V"):
        return {
            "price_source":"YAHOO_CHART_US_5M",
            "price_cadence_sec":300,
            "bar_timestamp_semantics":"OPEN",
            "official_exact_eligible":False,
        }
    if market=="KR" and normalized=="KR":
        return {
            "price_source":"DAUM_FINANCE_KR_5M",
            "price_cadence_sec":300,
            "bar_timestamp_semantics":"OPEN",
            "official_exact_eligible":False,
        }
    return {
        "price_source":"UNKNOWN",
        "price_cadence_sec":0,
        "bar_timestamp_semantics":"UNKNOWN",
        "official_exact_eligible":False,
    }

class PriceStore:
    def __init__(self,cfg:Config):
        self.cfg=cfg
        self.cache:Dict[Tuple[str,str],pd.DataFrame]={}
        self.kr_big=None

    def _individual_candidates(self,market,ticker):
        d=ROOT/(self.cfg.us_price_dir if market=="US" else self.cfg.kr_price_dir)
        return [
            d/f"{ticker}.parquet",
            d/f"{ticker}_clean.parquet",
            d/f"{ticker}.csv.gz",
            d/f"{ticker}.csv",
        ]

    def _download_hfdl(self,ticker:str)->Optional[Path]:
        key=os.getenv("HFDL_API_KEY","").strip()
        if not key:
            return None
        d=ROOT/self.cfg.us_price_dir
        d.mkdir(parents=True,exist_ok=True)
        p=d/f"{ticker}_clean.parquet"
        url=f"https://api.hfdatalibrary.com/v1/bars/{ticker}?version=clean"
        r=requests.get(url,headers={"X-API-Key":key},timeout=180)
        if r.status_code!=200:
            return None
        p.write_bytes(r.content)
        return p

    def load(self,market:str,ticker:str)->Optional[pd.DataFrame]:
        key=(market,ticker)
        if key in self.cache:
            return self.cache[key]

        for p in self._individual_candidates(market,ticker):
            if p.exists():
                df=pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
                z=normalize_bar_df(df,market,ticker)
                z.attrs.update(price_contract_metadata(p,market))
                self.cache[key]=z
                return z

        if market=="US":
            p=self._download_hfdl(ticker)
            if p and p.exists():
                z=normalize_bar_df(pd.read_parquet(p),market,ticker)
                z.attrs.update(price_contract_metadata(p,market))
                self.cache[key]=z
                return z

        # large KR parquet
        big=ROOT/self.cfg.kr_big_parquet
        if market=="KR" and big.exists():
            if self.kr_big is None:
                self.kr_big=pd.read_parquet(big)
                # normalize symbol
                symcol="symbol" if "symbol" in self.kr_big.columns else "ticker"
                self.kr_big[symcol]=self.kr_big[symcol].astype(str).str.zfill(6)
                self.kr_symcol=symcol
            part=self.kr_big[self.kr_big[self.kr_symcol]==ticker].copy()
            if not part.empty:
                z=normalize_bar_df(part,market,ticker)
                z.attrs.update({
                    "price_source":"KR_ALL_1M_UNAUDITED",
                    "price_cadence_sec":60,
                    "bar_timestamp_semantics":"UNKNOWN",
                    "official_exact_eligible":False,
                })
                self.cache[key]=z
                return z

        return None




def nearest_at_or_after(bars: pd.DataFrame, ts: pd.Timestamp) -> Optional[int]:
    a=bars["datetime"].to_numpy()
    i=int(np.searchsorted(a, np.datetime64(ts.tz_convert("UTC").tz_localize(None)), side="left"))
    # pandas tz -> numpy conversion above can be fragile; fallback
    if i>=len(bars):
        cand=bars.index[bars["datetime"]>=ts]
        return int(cand[0]) if len(cand) else None
    # use position
    return i

def bar_pos_at_or_after(bars:pd.DataFrame,ts:pd.Timestamp)->Optional[int]:
    # DatetimeArray.asi8 exposes the normalized nanosecond index without
    # allocating a full int64 copy for every event lookup.
    vals=bars["datetime"].array.asi8
    target=int(ts.value)
    i=int(np.searchsorted(vals,target,side="left"))
    return i if i<len(vals) else None


def source_family_for_event(ev:pd.Series)->str:
    market=str(ev.get("market","")).upper()
    source=str(ev.get("source","")).upper()
    event_id=str(ev.get("event_id","")).upper()
    if market=="US":
        return "US_SEC" if "SEC" in source or event_id.startswith("SEC:") else "US_NEWS"
    if market=="KR":
        return "KR_KIND" if "KIND" in source or event_id.startswith("KIND:") else "KR_NEWS"
    return "UNKNOWN"


def canonical_event_group_id(ev:pd.Series)->str:
    """Create a stable, outcome-free group key for split/overlap guards."""
    for column in ("accession_number","article_id","url","event_id"):
        value=str(ev.get(column,"") or "").strip()
        if value and value.lower()!="nan":
            payload=f"{ev.get('market','')}|{ev.get('ticker','')}|{column}|{value}"
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    payload=(f"{ev.get('market','')}|{ev.get('ticker','')}|"
             f"{ev.get('event_time_utc','')}|{ev.get('headline','')}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def event_to_row_exact_contract(
    ev:pd.Series,bars:pd.DataFrame,cfg:Config
)->Tuple[Optional[dict],str]:
    """Build a causal EXACT_T2_30M row from audited open-stamped <=1m bars.

    Entry and exit use the OPEN of the first bar at or after their respective
    targets.  For an open-stamped one-minute candle, its high/low/close/volume
    are not available at the timestamp, so every price-derived feature uses
    only bars whose end time is at or before the actual entry.
    """
    market=str(ev["market"])
    eutc=pd.Timestamp(ev["event_time_utc"])
    eutc=eutc.tz_localize("UTC") if eutc.tzinfo is None else eutc.tz_convert("UTC")
    metadata=dict(bars.attrs)
    cadence=int(metadata.get("price_cadence_sec",0) or 0)
    semantics=str(metadata.get("bar_timestamp_semantics","UNKNOWN")).upper()
    if not bool(metadata.get("official_exact_eligible",False)):
        return None,"NOT_EXACT_PRICE_ELIGIBLE"
    if cadence<=0 or cadence>cfg.max_official_cadence_sec:
        return None,"PRICE_CADENCE_INELIGIBLE"
    if semantics!="OPEN":
        return None,"BAR_SEMANTICS_UNKNOWN"

    local,op,cl=session_bounds(eutc,market)
    if not (op<=local<=cl):
        return None,"OFF_HOURS"
    entry_target=eutc+pd.Timedelta(minutes=cfg.entry_lag_min)
    ei=bar_pos_at_or_after(bars,entry_target)
    if ei is None:
        return None,"BAR_MISSING"
    actual_entry=pd.Timestamp(bars.iloc[ei]["datetime"]).tz_convert("UTC")
    entry_slippage=float((actual_entry-entry_target).total_seconds())
    if entry_slippage<0 or entry_slippage>cfg.max_entry_slippage_sec:
        return None,"ENTRY_TIMING_MISS"

    exit_target=actual_entry+pd.Timedelta(minutes=cfg.horizon_min)
    xi=bar_pos_at_or_after(bars,exit_target)
    if xi is None or xi<=ei:
        return None,"BAR_MISSING"
    actual_exit=pd.Timestamp(bars.iloc[xi]["datetime"]).tz_convert("UTC")
    exit_slippage=float((actual_exit-exit_target).total_seconds())
    hold_seconds=float((actual_exit-actual_entry).total_seconds())
    if exit_slippage<0 or exit_slippage>cfg.max_exit_slippage_sec:
        return None,"EXIT_TIMING_MISS"
    if not (cfg.min_hold_seconds<=hold_seconds<=cfg.max_hold_seconds):
        return None,"HOLD_TIMING_MISS"

    timezone=US_TZ if market=="US" else KR_TZ
    entry_local=actual_entry.tz_convert(timezone)
    exit_local=actual_exit.tz_convert(timezone)
    if entry_local.date()!=local.date() or exit_local.date()!=local.date():
        return None,"CROSSES_SESSION"
    if not (op<=entry_local<=cl and op<=exit_local<=cl):
        return None,"CROSSES_SESSION"

    entry_exit=pd.to_numeric(
        pd.Series([bars.iloc[ei]["open"],bars.iloc[xi]["open"]]),errors="coerce"
    )
    entry=float(entry_exit.iloc[0]);exitp=float(entry_exit.iloc[1])
    if not (np.isfinite(entry) and np.isfinite(exitp) and entry>0 and exitp>0):
        return None,"BAD_PRICE"

    # Only the current session prior to entry can contribute to a causal
    # clock-time feature.  Copying the symbol's complete multi-year 1-minute
    # history for every event is equivalent but prohibitively expensive.
    values=bars["datetime"].array.asi8
    session_start_utc=op.tz_convert("UTC")
    session_start=int(np.searchsorted(values,int(session_start_utc.value),side="left"))
    last_completed_open=actual_entry-pd.Timedelta(seconds=cadence)
    completed_end=int(np.searchsorted(values,int(last_completed_open.value),side="right"))
    work=bars.iloc[session_start:completed_end].copy()
    work["bar_end_utc"]=work["datetime"]+pd.to_timedelta(cadence,unit="s")
    completed=work[work["bar_end_utc"]<=actual_entry].copy()
    completed_local=completed["datetime"].dt.tz_convert(timezone)
    completed=completed[completed_local.dt.date==local.date()].copy()
    if len(completed)<2:
        return None,"NO_PREHISTORY"

    def clock_window(minutes:int)->pd.DataFrame:
        start=actual_entry-pd.Timedelta(minutes=minutes)
        return completed[(completed["datetime"]>=start)&
                         (completed["bar_end_utc"]<=actual_entry)].copy()

    def safe_returns(window:pd.DataFrame)->pd.Series:
        if len(window)<2:
            return pd.Series(dtype=float)
        return pd.to_numeric(window["close"],errors="coerce").pct_change().dropna()

    def asof_completed_close(target:pd.Timestamp)->float:
        eligible=completed[completed["bar_end_utc"]<=target]
        if eligible.empty:
            return np.nan
        row=eligible.iloc[-1]
        staleness=float((target-row["bar_end_utc"]).total_seconds())
        value=float(row["close"])
        if staleness<0 or staleness>cfg.max_feature_staleness_sec or value<=0:
            return np.nan
        return value

    def ret_back(minutes:int)->float:
        value=asof_completed_close(actual_entry-pd.Timedelta(minutes=minutes))
        return float(entry/value-1.0) if np.isfinite(value) and value>0 else np.nan

    def rolling_range(minutes:int)->float:
        window=clock_window(minutes)
        if window.empty:
            return np.nan
        high=pd.to_numeric(window["high"],errors="coerce").max()
        low=pd.to_numeric(window["low"],errors="coerce").min()
        return float((high-low)/entry) if np.isfinite(high) and np.isfinite(low) else np.nan

    def close_position(minutes:int)->float:
        window=clock_window(minutes)
        if window.empty:
            return np.nan
        high=float(pd.to_numeric(window["high"],errors="coerce").max())
        low=float(pd.to_numeric(window["low"],errors="coerce").min())
        return float((entry-low)/(high-low)) if np.isfinite(high) and high>low else 0.5

    def trend_slope(minutes:int)->float:
        window=clock_window(minutes)
        values=pd.to_numeric(window["close"],errors="coerce").to_numpy(float)
        valid=np.isfinite(values)&(values>0)
        if valid.sum()<5:
            return np.nan
        elapsed=(window["datetime"].astype("int64").to_numpy()-
                 int(window["datetime"].iloc[0].value))/float(pd.Timedelta(minutes=1).value)
        return float(np.polyfit(elapsed[valid],np.log(values[valid]),1)[0])

    def volume_mean(minutes:int)->float:
        values=pd.to_numeric(clock_window(minutes)["volume"],errors="coerce")
        return float(values.mean()) if len(values) and values.notna().any() else np.nan

    windows={minutes:clock_window(minutes) for minutes in (1,2,5,10,30,60)}
    returns={minutes:safe_returns(window) for minutes,window in windows.items()}
    vwap_window=windows[30]
    vwap_volume=pd.to_numeric(vwap_window["volume"],errors="coerce").fillna(0).to_numpy(float)
    vwap_close=pd.to_numeric(vwap_window["close"],errors="coerce").to_numpy(float)
    valid_vwap=np.isfinite(vwap_close)
    if valid_vwap.any() and vwap_volume[valid_vwap].sum()>0:
        vwap=float(np.average(vwap_close[valid_vwap],weights=vwap_volume[valid_vwap]))
    elif valid_vwap.any():
        vwap=float(np.nanmean(vwap_close))
    else:
        vwap=np.nan
    previous=completed.iloc[-1]
    previous_open=float(previous["open"]);previous_close=float(previous["close"])
    previous_high=float(previous["high"]);previous_low=float(previous["low"])
    session_open=float(completed.iloc[0]["open"])
    mean30=volume_mean(30);mean60=volume_mean(60)
    text=(str(ev.get("headline",""))+" "+str(ev.get("body",""))).strip()
    row={
        "event_id":ev["event_id"],"event_group_id":canonical_event_group_id(ev),
        "market":market,"ticker":ev["ticker"],"event_time_utc":eutc,
        "source":ev.get("source",""),"source_family":source_family_for_event(ev),
        "form":ev.get("form",""),"event_type":ev.get("event_type","OTHER"),
        "headline":ev.get("headline",""),"body":ev.get("body",""),"text":text,
        "contract_version":cfg.price_contract_version,
        "price_source":metadata.get("price_source","UNKNOWN"),
        "price_cadence_sec":cadence,"bar_timestamp_semantics":semantics,
        "entry_target_utc":entry_target,"actual_entry_time_utc":actual_entry,
        "entry_time_utc":actual_entry,"entry_slippage_sec":entry_slippage,
        "exit_target_utc":exit_target,"actual_exit_time_utc":actual_exit,
        "exit_time_utc":actual_exit,"exit_slippage_sec":exit_slippage,
        "actual_hold_seconds":hold_seconds,
        "entry_price":entry,"exit_price":exitp,"fwd_ret_30m":exitp/entry-1.0,
        "y":int(exitp>entry),
        **{f"pre_ret_{minutes}":ret_back(minutes)
           for minutes in (1,2,3,5,10,15,20,30,45,60,90,120)},
        **{f"pre_vol_{minutes}":float(returns[minutes].std())
           if len(returns[minutes])>=2 else np.nan for minutes in (5,10,30,60)},
        "volume_ratio_1_30":float(volume_mean(1)/max(mean30,1.0)) if np.isfinite(mean30) else np.nan,
        "volume_ratio_2_30":float(volume_mean(2)/max(mean30,1.0)) if np.isfinite(mean30) else np.nan,
        "volume_ratio_5_30":float(volume_mean(5)/max(mean30,1.0)) if np.isfinite(mean30) else np.nan,
        "volume_ratio_10_60":float(volume_mean(10)/max(mean60,1.0)) if np.isfinite(mean60) else np.nan,
        **{f"range_{minutes}m":rolling_range(minutes) for minutes in (5,10,30,60)},
        **{f"close_position_{minutes}":close_position(minutes) for minutes in (10,30,60)},
        "entry_bar_ret":float(previous_close/previous_open-1.0) if previous_open>0 else np.nan,
        "entry_bar_range":float((previous_high-previous_low)/previous_close) if previous_close>0 else np.nan,
        "intraday_ret_open":float(entry/session_open-1.0) if session_open>0 else np.nan,
        "return_autocorr_30":float(returns[30].autocorr(1)) if len(returns[30])>=3 else np.nan,
        "up_fraction_10":float((returns[10]>0).mean()) if len(returns[10]) else np.nan,
        "up_fraction_30":float((returns[30]>0).mean()) if len(returns[30]) else np.nan,
        "trend_slope_30":trend_slope(30),"trend_slope_60":trend_slope(60),
        "vwap_distance_30":float(entry/vwap-1.0) if np.isfinite(vwap) and vwap>0 else np.nan,
        "log_entry_price":float(np.log(entry)),
        "minutes_from_open":float((local-op).total_seconds()/60),
        "minutes_to_close":float((cl-local).total_seconds()/60),
        "event_positive_kw":count_kw(text,POSITIVE_KW),
        "event_negative_kw":count_kw(text,NEGATIVE_KW),
        "event_financing_kw":count_kw(text,FINANCING_KW),
        "event_trial_kw":count_kw(text,TRIAL_KW),
        "event_regulatory_kw":count_kw(text,REGULATORY_KW),
        "event_ma_kw":count_kw(text,MA_KW),"event_earnings_kw":count_kw(text,EARNINGS_KW),
        "headline_len":len(str(ev.get("headline",""))),"body_len":len(str(ev.get("body",""))),
    }
    return row,"OK"

def event_to_row(ev:pd.Series, bars:pd.DataFrame, cfg:Config)->Tuple[Optional[dict],str]:
    if cfg.official_exact_contract:
        return event_to_row_exact_contract(ev,bars,cfg)
    market=ev["market"]
    eutc=pd.Timestamp(ev["event_time_utc"])
    if eutc.tzinfo is None:
        eutc=eutc.tz_localize("UTC")
    else:
        eutc=eutc.tz_convert("UTC")

    local,op,cl=session_bounds(eutc,market)
    if not (op <= local <= cl):
        return None,"OFF_HOURS"
    # must leave room to exit
    if local + pd.Timedelta(minutes=cfg.entry_lag_min+cfg.horizon_min+1) > cl:
        return None,"TOO_LATE"

    entry_target=eutc+pd.Timedelta(minutes=cfg.entry_lag_min)
    exit_target=entry_target+pd.Timedelta(minutes=cfg.horizon_min)
    ei=bar_pos_at_or_after(bars,entry_target)
    xi=bar_pos_at_or_after(bars,exit_target)
    if ei is None or xi is None or xi<=ei:
        return None,"BAR_MISSING"

    # same local session
    et=bars.iloc[ei]["datetime"].tz_convert(US_TZ if market=="US" else KR_TZ)
    xt=bars.iloc[xi]["datetime"].tz_convert(US_TZ if market=="US" else KR_TZ)
    if et.date()!=local.date() or xt.date()!=local.date():
        return None,"CROSSES_SESSION"

    close=bars["close"].astype(float)
    entry=float(close.iloc[ei]); exitp=float(close.iloc[xi])
    if not (entry>0 and exitp>0):
        return None,"BAD_PRICE"

    def ret_back(minutes:int)->float:
        t=entry_target-pd.Timedelta(minutes=minutes)
        bi=bar_pos_at_or_after(bars,t)
        if bi is None or bi>=ei or close.iloc[bi]<=0:
            return np.nan
        return float(entry/close.iloc[bi]-1)

    pre=bars.iloc[max(0,ei-125):ei].copy()
    if len(pre)<10:
        return None,"NO_PREHISTORY"
    r=pre["close"].pct_change().dropna()
    v=pre["volume"].astype(float)
    entry_bar=bars.iloc[ei]
    entry_open=float(entry_bar.get("open",entry))
    entry_high=float(entry_bar.get("high",entry))
    entry_low=float(entry_bar.get("low",entry))
    session_history=bars.iloc[:ei+1].copy()
    session_local=pd.to_datetime(session_history["datetime"],utc=True).dt.tz_convert(
        US_TZ if market=="US" else KR_TZ
    )
    session_history=session_history[session_local.dt.date==local.date()]
    session_open=float(session_history.iloc[0]["open"]) if len(session_history) else entry

    def rolling_range(count:int)->float:
        window=pre.tail(count)
        if window.empty or not window["high"].notna().any():
            return np.nan
        return float((window["high"].max()-window["low"].min())/entry)

    def close_position(count:int)->float:
        window=pre.tail(count)
        if window.empty or not window["high"].notna().any():
            return np.nan
        low=float(window["low"].min()); high=float(window["high"].max())
        return float((entry-low)/(high-low)) if high>low else 0.5

    def trend_slope(count:int)->float:
        values=pd.to_numeric(pre["close"].tail(count),errors="coerce").dropna().to_numpy(float)
        if len(values)<5 or np.any(values<=0):
            return np.nan
        return float(np.polyfit(np.arange(len(values),dtype=float),np.log(values),1)[0])

    vwap_window=pre.tail(30)
    vwap_volume=pd.to_numeric(vwap_window["volume"],errors="coerce").fillna(0).to_numpy(float)
    vwap_close=pd.to_numeric(vwap_window["close"],errors="coerce").to_numpy(float)
    vwap=float(np.average(vwap_close,weights=vwap_volume)) if vwap_volume.sum()>0 else float(np.nanmean(vwap_close))

    text=(str(ev.get("headline",""))+" "+str(ev.get("body",""))).strip()
    row={
        "event_id":ev["event_id"],
        "market":market,
        "ticker":ev["ticker"],
        "event_time_utc":eutc,
        "source":ev.get("source",""),
        "event_type":ev.get("event_type","OTHER"),
        "headline":ev.get("headline",""),
        "body":ev.get("body",""),
        "text":text,
        "entry_time_utc":bars.iloc[ei]["datetime"],
        "exit_time_utc":bars.iloc[xi]["datetime"],
        "entry_price":entry,
        "exit_price":exitp,
        "fwd_ret_30m":exitp/entry-1,
        "y":int(exitp>entry),
        **{f"pre_ret_{minutes}":ret_back(minutes) for minutes in (1,2,3,5,10,15,20,30,45,60,90,120)},
        "pre_vol_5":float(r.tail(5).std()) if len(r)>=3 else np.nan,
        "pre_vol_10":float(r.tail(10).std()) if len(r)>=5 else np.nan,
        "pre_vol_30":float(r.tail(30).std()) if len(r)>=10 else np.nan,
        "pre_vol_60":float(r.tail(60).std()) if len(r)>=20 else np.nan,
        "volume_ratio_1_30":float(v.tail(1).mean()/max(v.tail(30).mean(),1.0)),
        "volume_ratio_2_30":float(v.tail(2).mean()/max(v.tail(30).mean(),1.0)),
        "volume_ratio_5_30":float(v.tail(5).mean()/max(v.tail(30).mean(),1.0)),
        "volume_ratio_10_60":float(v.tail(10).mean()/max(v.tail(60).mean(),1.0)),
        "range_5m":rolling_range(5),
        "range_10m":rolling_range(10),
        "range_30m":rolling_range(30),
        "range_60m":rolling_range(60),
        "close_position_10":close_position(10),
        "close_position_30":close_position(30),
        "close_position_60":close_position(60),
        "entry_bar_ret":float(entry/entry_open-1.0) if entry_open>0 else np.nan,
        "entry_bar_range":float((entry_high-entry_low)/entry) if entry>0 else np.nan,
        "intraday_ret_open":float(entry/session_open-1.0) if session_open>0 else np.nan,
        "return_autocorr_30":float(r.tail(30).autocorr(1)) if len(r)>=10 else np.nan,
        "up_fraction_10":float((r.tail(10)>0).mean()) if len(r)>=5 else np.nan,
        "up_fraction_30":float((r.tail(30)>0).mean()) if len(r)>=10 else np.nan,
        "trend_slope_30":trend_slope(30),
        "trend_slope_60":trend_slope(60),
        "vwap_distance_30":float(entry/vwap-1.0) if np.isfinite(vwap) and vwap>0 else np.nan,
        "log_entry_price":float(np.log(entry)),
        "minutes_from_open":float((local-op).total_seconds()/60),
        "minutes_to_close":float((cl-local).total_seconds()/60),
        "event_positive_kw":count_kw(text,POSITIVE_KW),
        "event_negative_kw":count_kw(text,NEGATIVE_KW),
        "event_financing_kw":count_kw(text,FINANCING_KW),
        "event_trial_kw":count_kw(text,TRIAL_KW),
        "event_regulatory_kw":count_kw(text,REGULATORY_KW),
        "event_ma_kw":count_kw(text,MA_KW),
        "event_earnings_kw":count_kw(text,EARNINGS_KW),
        "headline_len":len(str(ev.get("headline",""))),
        "body_len":len(str(ev.get("body",""))),
    }
    return row,"OK"

def add_benchmark_context(frame:pd.DataFrame,store:PriceStore)->pd.DataFrame:
    """Attach entry-time-only benchmark returns; never reads target returns."""
    out=frame.copy()
    for minutes in (2,5,15,30,60):
        out[f"benchmark_ret_{minutes}"]=np.nan
    benchmark={"US":"SPY","KR":"069500"}
    entry=pd.to_datetime(out["entry_time_utc"],utc=True)
    for market,ticker in benchmark.items():
        positions=np.flatnonzero(out["market"].astype(str).to_numpy()==market)
        if not len(positions):
            continue
        bars=store.load(market,ticker)
        if bars is None or bars.empty:
            raise RuntimeError(f"missing benchmark bars: {market}/{ticker}")
        bars=bars.sort_values("datetime").reset_index(drop=True)
        if store.cfg.official_exact_contract:
            metadata=dict(bars.attrs)
            cadence=int(metadata.get("price_cadence_sec",0) or 0)
            if (not metadata.get("official_exact_eligible",False) or
                    cadence<=0 or cadence>store.cfg.max_official_cadence_sec or
                    str(metadata.get("bar_timestamp_semantics","")).upper()!="OPEN"):
                raise RuntimeError(f"ineligible exact benchmark bars: {market}/{ticker}")
            bar_end=(bars["datetime"]+pd.to_timedelta(cadence,unit="s")).astype("int64").to_numpy()
            close=pd.to_numeric(bars["close"],errors="coerce").to_numpy(float)
            event_time=entry.iloc[positions].astype("int64").to_numpy()
            current=np.searchsorted(bar_end,event_time,side="right")-1
            for minutes in (2,5,15,30,60):
                target=event_time-int(pd.Timedelta(minutes=minutes).value)
                previous=np.searchsorted(bar_end,target,side="right")-1
                values=np.full(len(positions),np.nan,dtype=float)
                valid=(current>=0)&(previous>=0)
                current_safe=np.maximum(current,0);previous_safe=np.maximum(previous,0)
                valid&=(event_time-bar_end[current_safe])<=int(
                    pd.Timedelta(seconds=store.cfg.max_feature_staleness_sec).value
                )
                valid&=(target-bar_end[previous_safe])<=int(
                    pd.Timedelta(seconds=store.cfg.max_feature_staleness_sec).value
                )
                valid&=(close[current_safe]>0)&(close[previous_safe]>0)
                values[valid]=close[current_safe[valid]]/close[previous_safe[valid]]-1.0
                out.loc[out.index[positions],f"benchmark_ret_{minutes}"]=values
            continue
        bar_time=bars["datetime"].astype("int64").to_numpy()
        close=pd.to_numeric(bars["close"],errors="coerce").to_numpy(float)
        event_time=entry.iloc[positions].astype("int64").to_numpy()
        current=np.searchsorted(bar_time,event_time,side="right")-1
        for minutes in (2,5,15,30,60):
            previous=np.searchsorted(
                bar_time,event_time-int(pd.Timedelta(minutes=minutes).value),side="right"
            )-1
            values=np.full(len(positions),np.nan,dtype=float)
            valid=(current>=0)&(previous>=0)
            valid&=(event_time-bar_time[current])<=int(pd.Timedelta(minutes=10).value)
            valid&=(bar_time[current]-bar_time[previous])<=int(
                pd.Timedelta(minutes=minutes+10).value
            )
            valid&=(close[current]>0)&(close[previous]>0)
            values[valid]=close[current[valid]]/close[previous[valid]]-1.0
            out.loc[out.index[positions],f"benchmark_ret_{minutes}"]=values
    return out

def build_labeled(events:pd.DataFrame,cfg:Config)->pd.DataFrame:
    store=PriceStore(cfg)
    rows=[]; status={}
    for i,(_,ev) in enumerate(events.sort_values("event_time_utc").iterrows(),1):
        bars=store.load(ev["market"],ev["ticker"])
        if bars is None or bars.empty:
            st="NO_PRICE"
        else:
            row,st=event_to_row(ev,bars,cfg)
            if row is not None:
                rows.append(row)
        status[st]=status.get(st,0)+1
        if i%100==0:
            print(f"[LABEL] {i}/{len(events)} usable={len(rows)} status={status}")
    if not rows:
        raise RuntimeError(f"no labeled events. status={status}")
    x=pd.DataFrame(rows).sort_values("event_time_utc").reset_index(drop=True)
    x=add_benchmark_context(x,store)
    x.to_csv(LABELED_FILE,index=False,compression="gzip",encoding="utf-8")
    atomic_json(DATA/"LABEL_STATUS.json",status)
    return x




def model_specs(market:Optional[str]=None):
    # V35 is a pure transfer policy. Direction and confidence are selected on
    # opened V34, which is also the empirical-rank reference. V35 outcomes
    # remain unparsed until the one-shot seal.
    if market == "KR":
        return [{
            "kind":"v35_stable_rank","market":"KR",
            "reference_sources":["KR_EXACT_V34_SEAL"],
            "direction_cutoff":0.5000000000000002,
            "components":[
                ["pre_ret_45",-1.0,1.0/3.0],
                ["abs_pre_ret_30",1.0,1.0/3.0],
                ["intraday_ret_open",-1.0,1.0/3.0],
            ],
            "opportunity_column":"up_fraction_10","opportunity_sign":1.0,
            "margin_weight":0.00,"gate_quantile":0.650,
        }]
    if market == "US":
        return [{
            "kind":"v35_stable_rank","market":"US",
            "reference_sources":["US_EXACT_V34_SEAL"],
            "direction_cutoff":0.5400000000000003,
            "components":[
                ["return_autocorr_30",-1.0,1.0/3.0],
                ["volume_ratio_1_30",-1.0,1.0/3.0],
                ["event_positive_kw",-1.0,1.0/3.0],
            ],
            "opportunity_column":"abs_pre_ret_15","opportunity_sign":1.0,
            "margin_weight":0.00,"gate_quantile":0.850,
        }]
    out=[]
    for c in (0.15,0.35,0.7,1.5,3.0):
        for cw in (None,"balanced"):
            out.append({"kind":"logit","C":c,"class_weight":cw})
    for alpha in (1e-5,5e-5,2e-4):
        out.append({"kind":"sgd","alpha":alpha,"class_weight":"balanced"})
    return out

class NativeCatEstimator:
    """Small DataFrame-preserving wrapper for CatBoost categorical features."""
    def __init__(self,iterations:int,depth:int,cat_cols:Sequence[str]):
        self.iterations=iterations
        self.depth=depth
        self.cat_cols=list(cat_cols)
        self.numeric_cols=list(NUMERIC_COLS)
        self.columns=self.numeric_cols+self.cat_cols
        self.medians:Dict[str,float]={}
        self.model=None

    def _prepare(self,frame:pd.DataFrame,fit:bool=False)->pd.DataFrame:
        x=frame[self.columns].copy()
        for col in self.numeric_cols:
            values=pd.to_numeric(x[col],errors="coerce")
            if fit:
                median=float(values.median()) if values.notna().any() else 0.0
                self.medians[col]=median
            x[col]=values.fillna(self.medians.get(col,0.0))
        for col in self.cat_cols:
            x[col]=x[col].fillna("UNK").astype(str)
        return x

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        x=self._prepare(frame,fit=True)
        self.model=CatBoostClassifier(
            iterations=self.iterations,depth=self.depth,learning_rate=0.035,
            l2_leaf_reg=8.0,verbose=False,random_seed=SEED,
            auto_class_weights="Balanced",cat_features=self.cat_cols,
            thread_count=runtime_limits.THREAD_COUNT,
            task_type=runtime_limits.CATBOOST_TASK_TYPE,
            devices=runtime_limits.CATBOOST_DEVICES,allow_writing_files=False,
        )
        self.model.fit(x,np.asarray(y,dtype=int))
        return self

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        if self.model is None:
            raise RuntimeError("native CatBoost model is not fitted")
        return self.model.predict_proba(self._prepare(frame,fit=False))

class MeanReversionEstimator:
    """Label-free calibrated short-horizon reversal score for historical SEC."""
    def __init__(self,weights:Sequence[float],scale_quantile:float,scale_confidence:float,
                 columns:Sequence[str]=("pre_ret_2","pre_ret_5","pre_ret_15")):
        self.weights=tuple(float(value) for value in weights)
        self.columns=tuple(str(value) for value in columns)
        self.scale_quantile=float(scale_quantile)
        self.scale_confidence=float(scale_confidence)
        self.scale:Optional[float]=None

    def _score(self,frame:pd.DataFrame)->np.ndarray:
        values=frame.loc[:,self.columns].apply(pd.to_numeric,errors="coerce")
        # Match the pre-freeze DEV experiment: if any requested lookback is
        # unavailable (normally just after the open), emit a neutral score.
        weighted=sum(weight*values[column] for weight,column in zip(self.weights,self.columns))
        return -weighted.fillna(0).to_numpy(float)

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        score=self._score(frame)
        quantile=float(np.quantile(np.abs(score),self.scale_quantile))
        target_logit=math.log(self.scale_confidence/(1.0-self.scale_confidence))
        self.scale=max(quantile/max(target_logit,1e-6),1e-6)
        return self

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        if self.scale is None:
            raise RuntimeError("mean reversion estimator is not fitted")
        z=np.clip(self._score(frame)/self.scale,-30,30)
        p=1.0/(1.0+np.exp(-z))
        return np.column_stack([1.0-p,p])

class ProbabilityBlendEstimator:
    """DataFrame-preserving average of arbitrary local probability models."""
    def __init__(self,component_specs:Sequence[dict],weights:Sequence[float]):
        self.component_specs=[dict(spec) for spec in component_specs]
        self.weights=np.asarray(weights,dtype=float)
        self.models:list[Any]=[]

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        self.models=[]
        for spec in self.component_specs:
            model=make_model(spec)
            model.fit(frame,y)
            self.models.append(model)
        return self

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        if not self.models:
            raise RuntimeError("probability blend is not fitted")
        probabilities=np.stack([model.predict_proba(frame) for model in self.models])
        return np.average(probabilities,axis=0,weights=self.weights)

class KoreanRankBlendEstimator:
    """V13 Korean direction plus independently ranked confidence."""
    direction_cutoff=0.64
    category_weight=0.40

    def __init__(self,text_C:float=1.0,probability_scale:float=8.0):
        self.text_C=float(text_C)
        self.vectorizer:Optional[TfidfVectorizer]=None
        self.text_model:Optional[LogisticRegression]=None
        self.reference_scores:list[np.ndarray]=[]
        self.hour_mapping:Optional[pd.Series]=None
        self.hour_prior:Optional[float]=None
        self.category_reference:Optional[np.ndarray]=None
        self.distance_reference:Optional[np.ndarray]=None
        self.benchmark_reference:Optional[np.ndarray]=None
        self.confidence_reference:Optional[np.ndarray]=None

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        ranks=np.full(len(values),0.5,dtype=float)
        valid=np.isfinite(values)
        if len(reference):
            ranks[valid]=np.searchsorted(reference,values[valid],side="right")/len(reference)
        return ranks

    def _numeric_scores(self,frame:pd.DataFrame)->list[np.ndarray]:
        return [
            -pd.to_numeric(frame["pre_ret_5"],errors="coerce").fillna(0).to_numpy(float),
            -pd.to_numeric(frame["pre_ret_30"],errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(frame["pre_vol_10"],errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(frame["volume_ratio_5_30"],errors="coerce").fillna(0).to_numpy(float),
        ]

    @staticmethod
    def _hour_bin(frame:pd.DataFrame)->pd.Series:
        hour=pd.to_numeric(frame["event_hour"],errors="coerce").fillna(-1.0)
        return np.floor(hour*2.0).astype(int).astype(str)

    def _direction_components(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        if self.vectorizer is None or self.text_model is None or not self.reference_scores:
            raise RuntimeError("Korean direction estimator is not fitted")
        text_score=1.0-self.text_model.predict_proba(
            self.vectorizer.transform(frame[TEXT_COL].fillna(""))
        )[:,1]
        scores=[text_score]+self._numeric_scores(frame)
        base=np.mean([
            self._rank(reference,score)
            for reference,score in zip(self.reference_scores,scores)
        ],axis=0)
        if self.hour_mapping is None or self.hour_prior is None or self.category_reference is None:
            raise RuntimeError("Korean hour mapping is not fitted")
        category_raw=1.0-self._hour_bin(frame).map(self.hour_mapping).fillna(self.hour_prior).to_numpy(float)
        category=self._rank(self.category_reference,category_raw)
        return base,category

    def _confidence_quantile(self,frame:pd.DataFrame,direction:np.ndarray)->np.ndarray:
        if self.distance_reference is None or self.benchmark_reference is None or self.confidence_reference is None:
            raise RuntimeError("Korean confidence estimator is not fitted")
        distance=self._rank(self.distance_reference,np.abs(direction-self.direction_cutoff))
        benchmark=pd.to_numeric(frame["benchmark_ret_5"],errors="coerce").abs().to_numpy(float)
        benchmark_rank=self._rank(self.benchmark_reference,benchmark)
        raw=distance+1.4*benchmark_rank-0.4*(direction>=self.direction_cutoff)
        return self._rank(self.confidence_reference,raw)

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        self.vectorizer=TfidfVectorizer(
            lowercase=True,ngram_range=(1,2),min_df=2,max_features=50000,
            sublinear_tf=True,
        )
        text_matrix=self.vectorizer.fit_transform(frame[TEXT_COL].fillna(""))
        self.text_model=LogisticRegression(
            C=self.text_C,class_weight="balanced",max_iter=2000,
            solver="liblinear",random_state=SEED,
        )
        self.text_model.fit(text_matrix,np.asarray(y,dtype=int))
        labels=np.asarray(y,dtype=int)
        text_score=1.0-self.text_model.predict_proba(text_matrix)[:,1]
        scores=[text_score]+self._numeric_scores(frame)
        self.reference_scores=[
            np.sort(score[np.isfinite(score)]) for score in scores
        ]
        base=np.mean([
            self._rank(reference,score)
            for reference,score in zip(self.reference_scores,scores)
        ],axis=0)
        hour=self._hour_bin(frame)
        self.hour_prior=float(labels.mean())
        stats=pd.DataFrame({"hour":hour.to_numpy(),"y":labels}).groupby("hour").y.agg(["sum","count"])
        self.hour_mapping=(stats["sum"]+20.0*self.hour_prior)/(stats["count"]+20.0)
        category_raw=1.0-hour.map(self.hour_mapping).fillna(self.hour_prior).to_numpy(float)
        self.category_reference=np.sort(category_raw[np.isfinite(category_raw)])
        category=self._rank(self.category_reference,category_raw)
        direction=(1.0-self.category_weight)*base+self.category_weight*category
        distance_raw=np.abs(direction-self.direction_cutoff)
        self.distance_reference=np.sort(distance_raw[np.isfinite(distance_raw)])
        distance=self._rank(self.distance_reference,distance_raw)
        benchmark=pd.to_numeric(frame["benchmark_ret_5"],errors="coerce").abs().to_numpy(float)
        self.benchmark_reference=np.sort(benchmark[np.isfinite(benchmark)])
        benchmark_rank=self._rank(self.benchmark_reference,benchmark)
        confidence_raw=distance+1.4*benchmark_rank-0.4*(direction>=self.direction_cutoff)
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        return self

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        direction=self.direction_score(frame)
        confidence=self._confidence_quantile(frame,direction)
        probability=np.where(
            direction>=self.direction_cutoff,
            0.51+0.49*confidence,
            0.49-0.49*confidence,
        )
        return np.column_stack([1.0-probability,probability])

    def direction_score(self,frame:pd.DataFrame)->np.ndarray:
        base,category=self._direction_components(frame)
        return (1.0-self.category_weight)*base+self.category_weight*category

class USContextRuleEstimator:
    """V13 US market context and ticker/time target-rate blend."""
    columns=("benchmark_ret_30","pre_ret_60","minutes_from_open")
    signs=np.asarray([1.0,-1.0,-1.0])
    direction_cutoff=0.60
    category_weight=0.60

    def __init__(self):
        self.medians:Optional[pd.Series]=None
        self.means:Optional[pd.Series]=None
        self.stds:Optional[pd.Series]=None
        self.context_reference:Optional[np.ndarray]=None
        self.category_mapping:Optional[pd.Series]=None
        self.category_prior:Optional[float]=None
        self.category_reference:Optional[np.ndarray]=None
        self.distance_reference:Optional[np.ndarray]=None
        self.benchmark_reference:Optional[np.ndarray]=None
        self.confidence_reference:Optional[np.ndarray]=None

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        ranks=np.full(len(values),0.5,dtype=float)
        valid=np.isfinite(values)
        if len(reference):
            ranks[valid]=np.searchsorted(reference,values[valid],side="right")/len(reference)
        return ranks

    @staticmethod
    def _category(frame:pd.DataFrame)->pd.Series:
        hour=pd.to_numeric(frame["event_hour"],errors="coerce").fillna(-1.0)
        hour_bin=np.floor(hour*2.0).astype(int).astype(str)
        return frame["ticker"].fillna("UNK").astype(str)+"|"+hour_bin

    def _prepare(self,frame:pd.DataFrame)->pd.DataFrame:
        values=frame.loc[:,self.columns].apply(pd.to_numeric,errors="coerce")
        if self.medians is None:
            return values
        return values.fillna(self.medians)

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        values=self._prepare(frame)
        self.medians=values.median().fillna(0.0)
        values=values.fillna(self.medians)
        self.means=values.mean()
        self.stds=values.std().replace(0,np.nan).fillna(1.0)
        context=(((values-self.means)/self.stds)*self.signs).mean(axis=1).to_numpy(float)
        self.context_reference=np.sort(context[np.isfinite(context)])
        base=self._rank(self.context_reference,context)
        labels=np.asarray(y,dtype=int)
        category=self._category(frame)
        self.category_prior=float(labels.mean())
        stats=pd.DataFrame({"category":category.to_numpy(),"y":labels}).groupby("category").y.agg(["sum","count"])
        self.category_mapping=(stats["sum"]+2.0*self.category_prior)/(stats["count"]+2.0)
        category_raw=1.0-category.map(self.category_mapping).fillna(self.category_prior).to_numpy(float)
        self.category_reference=np.sort(category_raw[np.isfinite(category_raw)])
        category_rank=self._rank(self.category_reference,category_raw)
        direction=(1.0-self.category_weight)*base+self.category_weight*category_rank
        distance_raw=np.abs(direction-self.direction_cutoff)
        self.distance_reference=np.sort(distance_raw[np.isfinite(distance_raw)])
        distance=self._rank(self.distance_reference,distance_raw)
        benchmark=pd.to_numeric(frame["benchmark_ret_5"],errors="coerce").abs().to_numpy(float)
        self.benchmark_reference=np.sort(benchmark[np.isfinite(benchmark)])
        benchmark_rank=self._rank(self.benchmark_reference,benchmark)
        confidence_raw=distance+1.4*benchmark_rank-0.4*(direction>=self.direction_cutoff)
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        return self

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        direction=self.direction_score(frame)
        if self.distance_reference is None or self.benchmark_reference is None or self.confidence_reference is None:
            raise RuntimeError("US confidence mapping is not fitted")
        distance=self._rank(self.distance_reference,np.abs(direction-self.direction_cutoff))
        benchmark=pd.to_numeric(frame["benchmark_ret_5"],errors="coerce").abs().to_numpy(float)
        benchmark_rank=self._rank(self.benchmark_reference,benchmark)
        confidence_raw=distance+1.4*benchmark_rank-0.4*(direction>=self.direction_cutoff)
        confidence=0.5*self._rank(self.confidence_reference,confidence_raw)
        probability=np.where(
            direction>=self.direction_cutoff,
            0.51+0.49*confidence,
            0.49-0.49*confidence,
        )
        return np.column_stack([1.0-probability,probability])

    def direction_score(self,frame:pd.DataFrame)->np.ndarray:
        if self.medians is None or self.means is None or self.stds is None or self.context_reference is None:
            raise RuntimeError("US context rule is not fitted")
        values=self._prepare(frame)
        context=(((values-self.means)/self.stds)*self.signs).mean(axis=1).to_numpy(float)
        base=self._rank(self.context_reference,context)
        if self.category_mapping is None or self.category_prior is None or self.category_reference is None:
            raise RuntimeError("US category mapping is not fitted")
        category_raw=1.0-self._category(frame).map(self.category_mapping).fillna(self.category_prior).to_numpy(float)
        category=self._rank(self.category_reference,category_raw)
        return (1.0-self.category_weight)*base+self.category_weight*category

class FixedEntryRankModel:
    """One label-free technical/text rank model used inside V13 bagging."""
    KR_RULE=(
        ("pre_ret_20",-1.0),("pre_ret_5",-1.0),("entry_bar_ret",-1.0),
        ("benchmark_ret_60",-1.0),("pre_ret_15",-1.0),
    )
    US_RULE=(
        ("pre_ret_90",-1.0),("benchmark_ret_30",1.0),("trend_slope_30",1.0),
        ("volume_ratio_5_30",1.0),("event_positive_kw",1.0),
    )

    def __init__(self,market:str):
        self.market=str(market)
        self.rule=self.KR_RULE if self.market=="KR" else self.US_RULE
        self.references:Dict[str,np.ndarray]={}
        self.vectorizer:Optional[TfidfVectorizer]=None
        self.text_model:Optional[LogisticRegression]=None

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        output=np.full(len(values),0.5,dtype=float)
        valid=np.isfinite(values)
        if len(reference):
            output[valid]=np.searchsorted(reference,values[valid],side="right")/len(reference)
        return output

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        for column,sign in self.rule:
            values=sign*pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
            self.references[column]=np.sort(values[np.isfinite(values)])
        if self.market=="KR":
            self.vectorizer=TfidfVectorizer(
                lowercase=True,ngram_range=(1,2),min_df=2,
                max_features=50000,sublinear_tf=True,
            )
            matrix=self.vectorizer.fit_transform(frame[TEXT_COL].fillna(""))
            self.text_model=LogisticRegression(
                C=1.0,class_weight="balanced",max_iter=2000,
                solver="liblinear",random_state=SEED,
            )
            self.text_model.fit(matrix,np.asarray(y,dtype=int))
        return self

    def direction_score(self,frame:pd.DataFrame)->np.ndarray:
        components=[]
        for column,sign in self.rule:
            values=sign*pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
            components.append(self._rank(self.references[column],values))
        technical=np.mean(components,axis=0)
        if self.market!="KR":
            return technical
        if self.vectorizer is None or self.text_model is None:
            raise RuntimeError("Korean text rule is not fitted")
        inverse_text=1.0-self.text_model.predict_proba(
            self.vectorizer.transform(frame[TEXT_COL].fillna(""))
        )[:,1]
        return 0.6*technical+0.4*inverse_text

class RepeatedEntryRankEstimator:
    """Seven repeated grouped cross-fits with a frozen KR eligibility rule."""
    seeds=(260825,101,202,303,404,505,606)
    eligibility_direction_cutoff=0.675
    eligibility_quantile=0.75

    def __init__(self,market:str):
        self.market=str(market)
        self.models:list[FixedEntryRankModel]=[]
        self.oof_event_ids:Optional[np.ndarray]=None
        self.oof_direction:Optional[np.ndarray]=None
        self.full_reference:Optional[np.ndarray]=None
        self.oof_reference:Optional[np.ndarray]=None
        self.distance_reference:Optional[np.ndarray]=None
        self.eligibility_reference:Optional[np.ndarray]=None

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        return FixedEntryRankModel._rank(reference,values)

    @staticmethod
    def _quantile_map(reference_values:np.ndarray,quantiles:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference_values,dtype=float)[np.isfinite(reference_values)])
        if not len(reference):
            return np.full(len(quantiles),0.5,dtype=float)
        q=np.clip(np.asarray(quantiles,dtype=float),0.0,1.0)
        return np.quantile(reference,q,method="linear")

    def _groups(self,frame:pd.DataFrame)->pd.Series:
        if "event_id" not in frame or "event_time_utc" not in frame:
            raise RuntimeError("repeated estimator requires event_id and event_time_utc")
        if self.market=="KR":
            return frame["event_id"].astype(str).str.rsplit(":",n=1).str[-1]
        day=pd.to_datetime(frame["event_time_utc"],utc=True).dt.strftime("%Y-%m-%d")
        return frame["ticker"].fillna("UNK").astype(str)+"|"+day

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        labels=np.asarray(y,dtype=int)
        groups=self._groups(frame)
        oof_sum=np.zeros(len(frame),dtype=float)
        oof_count=np.zeros(len(frame),dtype=int)
        self.models=[]
        for seed in self.seeds:
            splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
            for train_index,valid_index in splitter.split(frame,labels,groups):
                model=FixedEntryRankModel(self.market)
                model.fit(frame.iloc[train_index],labels[train_index])
                oof_sum[valid_index]+=model.direction_score(frame.iloc[valid_index])
                oof_count[valid_index]+=1
                self.models.append(model)
        if (oof_count!=len(self.seeds)).any():
            raise RuntimeError("repeated cross-fit did not predict every row")
        self.oof_event_ids=frame["event_id"].astype(str).to_numpy()
        self.oof_direction=oof_sum/oof_count
        # The frozen ensemble uses all 28 training subsets. Quantile-map its
        # score back to the strictly cross-fitted DEV scale used for cutoffs.
        full_score=np.mean([model.direction_score(frame) for model in self.models],axis=0)
        self.full_reference=np.sort(full_score[np.isfinite(full_score)])
        self.oof_reference=np.sort(self.oof_direction[np.isfinite(self.oof_direction)])
        if self.market=="KR":
            distance_raw=np.abs(self.oof_direction-self.eligibility_direction_cutoff)
            self.distance_reference=np.sort(distance_raw[np.isfinite(distance_raw)])
            distance=self._rank(self.distance_reference,distance_raw)
            reliability=distance+0.5*(self.oof_direction>=self.eligibility_direction_cutoff)
            self.eligibility_reference=np.sort(reliability[np.isfinite(reliability)])
        return self

    def direction_score(self,frame:pd.DataFrame)->np.ndarray:
        if not self.models or self.full_reference is None or self.oof_reference is None:
            raise RuntimeError("repeated entry rank estimator is not fitted")
        raw=np.mean([model.direction_score(frame) for model in self.models],axis=0)
        percentile=self._rank(self.full_reference,raw)
        return self._quantile_map(self.oof_reference,percentile)

    def eligible_from_direction(self,direction:np.ndarray)->np.ndarray:
        if self.market=="US":
            return np.ones(len(direction),dtype=bool)
        if self.distance_reference is None or self.eligibility_reference is None:
            raise RuntimeError("Korean eligibility rule is not fitted")
        distance=self._rank(
            self.distance_reference,
            np.abs(np.asarray(direction,dtype=float)-self.eligibility_direction_cutoff),
        )
        reliability=distance+0.5*(np.asarray(direction)>=self.eligibility_direction_cutoff)
        quantile=self._rank(self.eligibility_reference,reliability)
        return quantile>=self.eligibility_quantile

    def eligible(self,frame:pd.DataFrame)->np.ndarray:
        return self.eligible_from_direction(self.direction_score(frame))

    def crossfit_output(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        if self.oof_event_ids is None or self.oof_direction is None:
            raise RuntimeError("cross-fit output unavailable")
        event_ids=frame["event_id"].astype(str).to_numpy()
        if not np.array_equal(event_ids,self.oof_event_ids):
            lookup={event_id:index for index,event_id in enumerate(self.oof_event_ids)}
            try:
                direction=np.asarray([self.oof_direction[lookup[event_id]] for event_id in event_ids])
            except KeyError as exc:
                raise RuntimeError(f"cross-fit event mismatch: {exc}") from exc
        else:
            direction=self.oof_direction.copy()
        return direction,self.eligible_from_direction(direction)

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        probability=np.clip(self.direction_score(frame),1e-6,1-1e-6)
        return np.column_stack([1.0-probability,probability])

class V14RepeatedOpportunityEstimator:
    """Repeated grouped direction ensemble with entry-time opportunity confidence."""
    seeds=(260825,101,202,303,404)

    def __init__(self,market:str,direction_cutoff:float,opportunity_column:str,
                 opportunity_sign:float,margin_weight:float=0.25):
        self.market=str(market)
        self.direction_cutoff=float(direction_cutoff)
        self.opportunity_column=str(opportunity_column)
        self.opportunity_sign=float(opportunity_sign)
        self.margin_weight=float(margin_weight)
        self.model_groups:list[list[Any]]=[]
        self.oof_event_ids:Optional[np.ndarray]=None
        self.oof_direction:Optional[np.ndarray]=None
        self.oof_probability:Optional[np.ndarray]=None
        self.full_reference:Optional[np.ndarray]=None
        self.oof_reference:Optional[np.ndarray]=None
        self.opportunity_reference:Optional[np.ndarray]=None
        self.margin_reference:Optional[np.ndarray]=None
        self.confidence_reference:Optional[np.ndarray]=None

    def _base_specs(self)->list[dict]:
        if self.market=="KR":
            return [{"kind":"cat","iterations":120,"depth":3}]
        return [
            {"kind":"native_cat","iterations":150,"depth":3,
             "cat_cols":["event_type","ticker"]},
            {"kind":"lgb","n_estimators":120,"num_leaves":7},
            {"kind":"lgb","n_estimators":80,"num_leaves":3},
            {"kind":"logit","C":1.0,"class_weight":"balanced"},
        ]

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        output=np.full(len(values),0.5,dtype=float)
        valid=np.isfinite(values)
        if len(reference):
            output[valid]=np.searchsorted(reference,values[valid],side="right")/len(reference)
        return output

    @staticmethod
    def _quantile_map(reference_values:np.ndarray,quantiles:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference_values,dtype=float)[np.isfinite(reference_values)])
        if not len(reference):
            return np.full(len(quantiles),0.5,dtype=float)
        return np.quantile(reference,np.clip(quantiles,0.0,1.0),method="linear")

    def _groups(self,frame:pd.DataFrame)->pd.Series:
        if self.market=="KR":
            return frame["event_id"].astype(str).str.rsplit(":",n=1).str[-1]
        day=pd.to_datetime(frame["event_time_utc"],utc=True).dt.strftime("%Y-%m-%d")
        return frame["ticker"].fillna("UNK").astype(str)+"|"+day

    def _opportunity(self,frame:pd.DataFrame)->np.ndarray:
        return self.opportunity_sign*pd.to_numeric(
            frame[self.opportunity_column],errors="coerce"
        ).to_numpy(float)

    def _confidence(self,frame:pd.DataFrame,direction:np.ndarray)->np.ndarray:
        if (self.opportunity_reference is None or self.margin_reference is None or
                self.confidence_reference is None):
            raise RuntimeError("V14 opportunity confidence is not fitted")
        opportunity=self._rank(self.opportunity_reference,self._opportunity(frame))
        margin=self._rank(self.margin_reference,np.abs(direction-self.direction_cutoff))
        raw=self.margin_weight*margin+(1.0-self.margin_weight)*opportunity
        opportunity_score=self._rank(self.confidence_reference,raw)
        # Preserve the model-margin ordering for ordinary observations while
        # reserving only the entry-time opportunity top 20% for certification
        # confidence.  This keeps high-confidence economics from distorting
        # the full-sample AUC ranking.
        return np.where(
            opportunity_score>=0.80,
            0.86+0.14*margin,
            0.84*margin,
        )

    def _probability(self,frame:pd.DataFrame,direction:np.ndarray)->np.ndarray:
        confidence=self._confidence(frame,direction)
        return np.where(
            direction>=self.direction_cutoff,
            0.50+0.50*confidence,
            0.50-0.50*confidence,
        )

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        labels=np.asarray(y,dtype=int)
        groups=self._groups(frame)
        oof_sum=np.zeros(len(frame),dtype=float)
        oof_count=np.zeros(len(frame),dtype=int)
        self.model_groups=[]
        for seed in self.seeds:
            splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
            for train_index,valid_index in splitter.split(frame,labels,groups):
                models=[];fold_predictions=[]
                for spec in self._base_specs():
                    model=make_model(spec)
                    model.fit(frame.iloc[train_index],labels[train_index])
                    fold_predictions.append(model.predict_proba(frame.iloc[valid_index])[:,1])
                    models.append(model)
                oof_sum[valid_index]+=np.mean(fold_predictions,axis=0)
                oof_count[valid_index]+=1
                self.model_groups.append(models)
        if (oof_count!=len(self.seeds)).any():
            raise RuntimeError("V14 repeated cross-fit did not predict every row")
        self.oof_event_ids=frame["event_id"].astype(str).to_numpy()
        self.oof_direction=oof_sum/oof_count
        full_score=np.mean([
            np.mean([model.predict_proba(frame)[:,1] for model in models],axis=0)
            for models in self.model_groups
        ],axis=0)
        self.full_reference=np.sort(full_score[np.isfinite(full_score)])
        self.oof_reference=np.sort(self.oof_direction[np.isfinite(self.oof_direction)])
        opportunity=self._opportunity(frame)
        self.opportunity_reference=np.sort(opportunity[np.isfinite(opportunity)])
        margin_raw=np.abs(self.oof_direction-self.direction_cutoff)
        self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])
        opportunity_rank=self._rank(self.opportunity_reference,opportunity)
        margin_rank=self._rank(self.margin_reference,margin_raw)
        confidence_raw=(
            self.margin_weight*margin_rank+(1.0-self.margin_weight)*opportunity_rank
        )
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        self.oof_probability=self._probability(frame,self.oof_direction)
        return self

    def direction_score(self,frame:pd.DataFrame)->np.ndarray:
        if not self.model_groups or self.full_reference is None or self.oof_reference is None:
            raise RuntimeError("V14 repeated estimator is not fitted")
        raw=np.mean([
            np.mean([model.predict_proba(frame)[:,1] for model in models],axis=0)
            for models in self.model_groups
        ],axis=0)
        percentile=self._rank(self.full_reference,raw)
        return self._quantile_map(self.oof_reference,percentile)

    def crossfit_output(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        if self.oof_event_ids is None or self.oof_probability is None:
            raise RuntimeError("V14 cross-fit output unavailable")
        event_ids=frame["event_id"].astype(str).to_numpy()
        if np.array_equal(event_ids,self.oof_event_ids):
            probability=self.oof_probability.copy()
        else:
            lookup={event_id:index for index,event_id in enumerate(self.oof_event_ids)}
            try:
                probability=np.asarray([self.oof_probability[lookup[event_id]] for event_id in event_ids])
            except KeyError as exc:
                raise RuntimeError(f"V14 cross-fit event mismatch: {exc}") from exc
        return probability,np.ones(len(frame),dtype=bool)

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        direction=self.direction_score(frame)
        probability=np.clip(self._probability(frame,direction),1e-6,1-1e-6)
        return np.column_stack([1.0-probability,probability])

class V15RepeatedOpportunityEstimator:
    """V15 repeated grouped ensemble and frozen entry-time confidence gate."""
    seeds=(260825,101,202,303,404)

    def __init__(self,market:str,direction_cutoff:float,opportunity_column:str,
                 opportunity_sign:float,margin_weight:float,gate_quantile:float):
        self.market=str(market)
        self.direction_cutoff=float(direction_cutoff)
        self.opportunity_column=str(opportunity_column)
        self.opportunity_sign=float(opportunity_sign)
        self.margin_weight=float(margin_weight)
        self.gate_quantile=float(gate_quantile)
        self.models:list[Any]=[]
        self.oof_event_ids:Optional[np.ndarray]=None
        self.oof_direction:Optional[np.ndarray]=None
        self.oof_probability:Optional[np.ndarray]=None
        self.full_reference:Optional[np.ndarray]=None
        self.oof_reference:Optional[np.ndarray]=None
        self.opportunity_reference:Optional[np.ndarray]=None
        self.margin_reference:Optional[np.ndarray]=None
        self.confidence_reference:Optional[np.ndarray]=None

    def _base_spec(self)->dict:
        if self.market=="US":
            return {
                "kind":"native_cat","iterations":150,"depth":3,
                "cat_cols":["event_type","ticker"],
            }
        if self.market=="KR":
            return {"kind":"logit","C":0.01,"class_weight":"balanced"}
        raise RuntimeError(f"unsupported V15 market: {self.market}")

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        output=np.full(len(values),0.5,dtype=float)
        valid=np.isfinite(values)
        if len(reference):
            output[valid]=np.searchsorted(reference,values[valid],side="right")/len(reference)
        return output

    @staticmethod
    def _quantile_map(reference_values:np.ndarray,quantiles:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference_values,dtype=float)[np.isfinite(reference_values)])
        if not len(reference):
            return np.full(len(quantiles),0.5,dtype=float)
        return np.quantile(reference,np.clip(quantiles,0.0,1.0),method="linear")

    @staticmethod
    def _groups(frame:pd.DataFrame)->pd.Series:
        day=pd.to_datetime(frame["event_time_utc"],utc=True).dt.strftime("%Y-%m-%d")
        return frame["ticker"].fillna("UNK").astype(str)+"|"+day

    def _opportunity(self,frame:pd.DataFrame)->np.ndarray:
        return self.opportunity_sign*pd.to_numeric(
            frame[self.opportunity_column],errors="coerce"
        ).to_numpy(float)

    def _confidence(self,frame:pd.DataFrame,direction:np.ndarray)->np.ndarray:
        if (self.opportunity_reference is None or self.margin_reference is None or
                self.confidence_reference is None):
            raise RuntimeError("V15 opportunity confidence is not fitted")
        opportunity=self._rank(self.opportunity_reference,self._opportunity(frame))
        margin=self._rank(self.margin_reference,np.abs(direction-self.direction_cutoff))
        raw=self.margin_weight*margin+(1.0-self.margin_weight)*opportunity
        opportunity_score=self._rank(self.confidence_reference,raw)
        return np.where(
            opportunity_score>=self.gate_quantile,
            0.86+0.14*margin,
            0.84*margin,
        )

    def _probability(self,frame:pd.DataFrame,direction:np.ndarray)->np.ndarray:
        confidence=self._confidence(frame,direction)
        return np.where(
            direction>=self.direction_cutoff,
            0.50+0.50*confidence,
            0.50-0.50*confidence,
        )

    def fit_with_prior(self,frame:pd.DataFrame,y:Sequence[int],
                       prior_frame:Optional[pd.DataFrame]=None,
                       prior_y:Optional[Sequence[int]]=None):
        labels=np.asarray(y,dtype=int)
        groups=self._groups(frame)
        use_prior=prior_frame is not None and prior_y is not None and len(prior_frame)>0
        prior_labels=np.asarray(prior_y,dtype=int) if use_prior else np.empty(0,dtype=int)
        oof_sum=np.zeros(len(frame),dtype=float)
        oof_count=np.zeros(len(frame),dtype=int)
        self.models=[]
        for seed in self.seeds:
            splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
            for train_index,valid_index in splitter.split(frame,labels,groups):
                train_frame=frame.iloc[train_index]
                train_labels=labels[train_index]
                if use_prior:
                    train_frame=pd.concat([prior_frame,train_frame],ignore_index=True)
                    train_labels=np.concatenate([prior_labels,train_labels])
                model=make_model(self._base_spec())
                model.fit(train_frame,train_labels)
                oof_sum[valid_index]+=model.predict_proba(frame.iloc[valid_index])[:,1]
                oof_count[valid_index]+=1
                self.models.append(model)
        if (oof_count!=len(self.seeds)).any():
            raise RuntimeError("V15 repeated cross-fit did not predict every row")
        self.oof_event_ids=frame["event_id"].astype(str).to_numpy()
        self.oof_direction=oof_sum/oof_count
        full_score=np.mean(
            [model.predict_proba(frame)[:,1] for model in self.models],axis=0
        )
        self.full_reference=np.sort(full_score[np.isfinite(full_score)])
        self.oof_reference=np.sort(self.oof_direction[np.isfinite(self.oof_direction)])
        opportunity=self._opportunity(frame)
        self.opportunity_reference=np.sort(opportunity[np.isfinite(opportunity)])
        margin_raw=np.abs(self.oof_direction-self.direction_cutoff)
        self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])
        opportunity_rank=self._rank(self.opportunity_reference,opportunity)
        margin_rank=self._rank(self.margin_reference,margin_raw)
        confidence_raw=(
            self.margin_weight*margin_rank+(1.0-self.margin_weight)*opportunity_rank
        )
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        self.oof_probability=self._probability(frame,self.oof_direction)
        return self

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        return self.fit_with_prior(frame,y)

    def direction_score(self,frame:pd.DataFrame)->np.ndarray:
        if not self.models or self.full_reference is None or self.oof_reference is None:
            raise RuntimeError("V15 repeated estimator is not fitted")
        raw=np.mean([model.predict_proba(frame)[:,1] for model in self.models],axis=0)
        percentile=self._rank(self.full_reference,raw)
        return self._quantile_map(self.oof_reference,percentile)

    def crossfit_output(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        if self.oof_event_ids is None or self.oof_probability is None:
            raise RuntimeError("V15 cross-fit output unavailable")
        event_ids=frame["event_id"].astype(str).to_numpy()
        if np.array_equal(event_ids,self.oof_event_ids):
            probability=self.oof_probability.copy()
        else:
            lookup={event_id:index for index,event_id in enumerate(self.oof_event_ids)}
            try:
                probability=np.asarray([self.oof_probability[lookup[event_id]] for event_id in event_ids])
            except KeyError as exc:
                raise RuntimeError(f"V15 cross-fit event mismatch: {exc}") from exc
        return probability,np.ones(len(frame),dtype=bool)

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        direction=self.direction_score(frame)
        probability=np.clip(self._probability(frame,direction),1e-6,1-1e-6)
        return np.column_stack([1.0-probability,probability])

class V17StableRankEstimator:
    """Frozen entry-time rank direction and confidence policy for V17."""
    def __init__(self,market:str,direction_cutoff:float,components:Sequence[Sequence[Any]],
                 opportunity_column:str,opportunity_sign:float,margin_weight:float,
                 gate_quantile:float):
        self.market=str(market)
        self.direction_cutoff=float(direction_cutoff)
        self.components=[(str(c),float(s),float(w)) for c,s,w in components]
        self.opportunity_column=str(opportunity_column)
        self.opportunity_sign=float(opportunity_sign)
        self.margin_weight=float(margin_weight)
        self.gate_quantile=float(gate_quantile)
        self.medians:dict[str,float]={}
        self.references:dict[str,np.ndarray]={}
        self.margin_reference:Optional[np.ndarray]=None
        self.opportunity_reference:Optional[np.ndarray]=None
        self.confidence_reference:Optional[np.ndarray]=None
        self.oof_event_ids:Optional[np.ndarray]=None
        self.oof_probability:Optional[np.ndarray]=None

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        if not len(reference):
            return np.full(len(values),0.5,dtype=float)
        return np.searchsorted(reference,values,side="right")/len(reference)

    def _values(self,frame:pd.DataFrame,column:str,fit:bool=False)->np.ndarray:
        values=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
        if fit:
            median=float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            self.medians[column]=median
        fill=self.medians.get(column,0.0)
        return np.nan_to_num(values,nan=fill,posinf=fill,neginf=fill)

    def direction_score(self,frame:pd.DataFrame)->np.ndarray:
        if not self.references:
            raise RuntimeError("V17 stable rank estimator is not fitted")
        pieces=[];weights=[]
        for column,sign,weight in self.components:
            values=self._values(frame,column)
            component=self._rank(self.references[column],values)
            pieces.append(weight*(component if sign>0.0 else 1.0-component))
            weights.append(weight)
        return np.sum(pieces,axis=0)/sum(weights)

    def _confidence(self,frame:pd.DataFrame,direction:np.ndarray)->np.ndarray:
        if (self.margin_reference is None or self.opportunity_reference is None or
                self.confidence_reference is None):
            raise RuntimeError("V17 confidence policy is not fitted")
        margin=self._rank(
            self.margin_reference,np.abs(direction-self.direction_cutoff)
        )
        opportunity=self._rank(
            self.opportunity_reference,
            self.opportunity_sign*self._values(frame,self.opportunity_column),
        )
        raw=self.margin_weight*margin+(1.0-self.margin_weight)*opportunity
        opportunity_score=self._rank(self.confidence_reference,raw)
        return np.where(
            opportunity_score>=self.gate_quantile,
            0.86+0.14*margin,
            0.84*margin,
        )

    def _probability(self,frame:pd.DataFrame,direction:np.ndarray)->np.ndarray:
        confidence=self._confidence(frame,direction)
        return np.where(
            direction>=self.direction_cutoff,
            0.50+0.50*confidence,
            0.50-0.50*confidence,
        )

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        del y  # V17 ranks are label-independent after their frozen DEV selection.
        columns={column for column,_,_ in self.components}|{self.opportunity_column}
        self.references={}
        for column in sorted(columns):
            values=self._values(frame,column,fit=True)
            self.references[column]=np.sort(values[np.isfinite(values)])
        direction=self.direction_score(frame)
        margin_raw=np.abs(direction-self.direction_cutoff)
        self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])
        opportunity=self.opportunity_sign*self._values(frame,self.opportunity_column)
        self.opportunity_reference=np.sort(opportunity[np.isfinite(opportunity)])
        margin=self._rank(self.margin_reference,margin_raw)
        opportunity_rank=self._rank(self.opportunity_reference,opportunity)
        confidence_raw=(
            self.margin_weight*margin+(1.0-self.margin_weight)*opportunity_rank
        )
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        self.oof_event_ids=frame["event_id"].astype(str).to_numpy()
        self.oof_probability=self._probability(frame,direction)
        return self

    def crossfit_output(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        if self.oof_event_ids is None or self.oof_probability is None:
            raise RuntimeError("V17 fitted DEV output unavailable")
        event_ids=frame["event_id"].astype(str).to_numpy()
        if np.array_equal(event_ids,self.oof_event_ids):
            probability=self.oof_probability.copy()
        else:
            lookup={event_id:index for index,event_id in enumerate(self.oof_event_ids)}
            try:
                probability=np.asarray([self.oof_probability[lookup[event_id]] for event_id in event_ids])
            except KeyError as exc:
                raise RuntimeError(f"V17 DEV event mismatch: {exc}") from exc
        return probability,np.ones(len(frame),dtype=bool)

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        direction=self.direction_score(frame)
        probability=np.clip(self._probability(frame,direction),1e-6,1-1e-6)
        return np.column_stack([1.0-probability,probability])

class V18PriorTransferEstimator:
    """Prior-only direction model with V18 DEV-ranked confidence."""
    def __init__(self,market:str,base_spec:dict,raw_cutoff:float,
                 opportunity_column:str,opportunity_sign:float,
                 margin_weight:float,gate_quantile:float):
        self.market=str(market)
        self.base_spec=dict(base_spec)
        self.raw_cutoff=float(raw_cutoff)
        self.opportunity_column=str(opportunity_column)
        self.opportunity_sign=float(opportunity_sign)
        self.margin_weight=float(margin_weight)
        self.gate_quantile=float(gate_quantile)
        self.base_model:Optional[Any]=None
        self.opportunity_median:float=0.0
        self.margin_reference:Optional[np.ndarray]=None
        self.opportunity_reference:Optional[np.ndarray]=None
        self.confidence_reference:Optional[np.ndarray]=None
        self.oof_event_ids:Optional[np.ndarray]=None
        self.oof_probability:Optional[np.ndarray]=None

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        if not len(reference):
            return np.full(len(values),0.5,dtype=float)
        return np.searchsorted(reference,values,side="right")/len(reference)

    def _opportunity(self,frame:pd.DataFrame,fit:bool=False)->np.ndarray:
        values=pd.to_numeric(frame[self.opportunity_column],errors="coerce").to_numpy(float)
        if fit:
            self.opportunity_median=(
                float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            )
        values=np.nan_to_num(
            values,nan=self.opportunity_median,
            posinf=self.opportunity_median,neginf=self.opportunity_median,
        )
        return self.opportunity_sign*values

    def _probability(self,frame:pd.DataFrame,raw:np.ndarray)->np.ndarray:
        if (self.margin_reference is None or self.opportunity_reference is None or
                self.confidence_reference is None):
            raise RuntimeError("V18 transfer confidence policy is not fitted")
        margin=self._rank(self.margin_reference,np.abs(raw-self.raw_cutoff))
        opportunity=self._rank(self.opportunity_reference,self._opportunity(frame))
        score=self._rank(
            self.confidence_reference,
            self.margin_weight*margin+(1.0-self.margin_weight)*opportunity,
        )
        high=score>=self.gate_quantile
        confidence=np.where(high,0.86+0.14*margin,0.84*margin)
        return np.where(
            raw>=self.raw_cutoff,
            0.50+0.50*confidence,
            0.50-0.50*confidence,
        )

    def fit_with_prior(self,frame:pd.DataFrame,y:Sequence[int],
                       prior_frame:pd.DataFrame,prior_y:Sequence[int]):
        del y  # V18 target labels remain external to the direction fit.
        if prior_frame is None or prior_y is None or len(prior_frame)==0:
            raise RuntimeError("V18 transfer model requires opened prior rows")
        tickers=set(frame["ticker"].astype(str))
        eligible=prior_frame["ticker"].astype(str).isin(tickers).to_numpy()
        if not eligible.any():
            raise RuntimeError("V18 transfer model has no same-ticker prior rows")
        fit_frame=prior_frame.loc[eligible].copy()
        fit_y=np.asarray(prior_y,dtype=int)[eligible]
        if len(np.unique(fit_y))<2:
            raise RuntimeError("V18 same-ticker prior labels are one-class")
        self.base_model=make_model(self.base_spec)
        self.base_model.fit(fit_frame,fit_y)
        raw=self.base_model.predict_proba(frame)[:,1]
        margin_raw=np.abs(raw-self.raw_cutoff)
        self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])
        opportunity=self._opportunity(frame,fit=True)
        self.opportunity_reference=np.sort(opportunity[np.isfinite(opportunity)])
        margin=self._rank(self.margin_reference,margin_raw)
        opportunity_rank=self._rank(self.opportunity_reference,opportunity)
        confidence_raw=(
            self.margin_weight*margin+(1.0-self.margin_weight)*opportunity_rank
        )
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        self.oof_event_ids=frame["event_id"].astype(str).to_numpy()
        self.oof_probability=self._probability(frame,raw)
        return self

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        del frame,y
        raise RuntimeError("use fit_with_prior for V18 transfer estimator")

    def crossfit_output(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        if self.oof_event_ids is None or self.oof_probability is None:
            raise RuntimeError("V18 external DEV output unavailable")
        event_ids=frame["event_id"].astype(str).to_numpy()
        lookup={event_id:index for index,event_id in enumerate(self.oof_event_ids)}
        try:
            probability=np.asarray([self.oof_probability[lookup[event_id]] for event_id in event_ids])
        except KeyError as exc:
            raise RuntimeError(f"V18 DEV event mismatch: {exc}") from exc
        return probability,np.ones(len(frame),dtype=bool)

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        if self.base_model is None:
            raise RuntimeError("V18 transfer estimator is not fitted")
        raw=self.base_model.predict_proba(frame)[:,1]
        probability=np.clip(self._probability(frame,raw),1e-6,1-1e-6)
        return np.column_stack([1.0-probability,probability])

class V19StableRankEstimator(V17StableRankEstimator):
    """V19 frozen DEV-rank policy; inherited mechanics are outcome-free."""

class V20StableRankEstimator(V17StableRankEstimator):
    """V20 DEV-rank policy with exact empirical midranks for tied features."""

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        if not len(reference):
            return np.full(len(values),0.5,dtype=float)
        left=np.searchsorted(reference,values,side="left")
        right=np.searchsorted(reference,values,side="right")
        return (left+right+1.0)/(2.0*len(reference))

class V21StableRankEstimator(V20StableRankEstimator):
    """V21 multi-audit rank policy."""

class V21SupervisedConfidenceEstimator:
    """Grouped-CV direction model with entry-time rank confidence gating."""
    def __init__(self,market:str,base_spec:dict,raw_cutoff:float,
                 opportunity_column:str,opportunity_sign:float,
                 margin_weight:float,gate_quantile:float):
        self.market=str(market);self.base_spec=dict(base_spec)
        self.raw_cutoff=float(raw_cutoff);self.opportunity_column=str(opportunity_column)
        self.opportunity_sign=float(opportunity_sign);self.margin_weight=float(margin_weight)
        self.gate_quantile=float(gate_quantile);self.base_model:Optional[Any]=None
        self.opportunity_median=0.0;self.margin_reference:Optional[np.ndarray]=None
        self.opportunity_reference:Optional[np.ndarray]=None
        self.confidence_reference:Optional[np.ndarray]=None

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        return V20StableRankEstimator._rank(reference,values)

    def _opportunity(self,frame:pd.DataFrame,fit:bool=False)->np.ndarray:
        values=pd.to_numeric(frame[self.opportunity_column],errors="coerce").to_numpy(float)
        if fit:self.opportunity_median=float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        return self.opportunity_sign*np.nan_to_num(values,nan=self.opportunity_median,
                                                   posinf=self.opportunity_median,neginf=self.opportunity_median)

    def _probability(self,frame:pd.DataFrame,raw:np.ndarray)->np.ndarray:
        if self.margin_reference is None or self.opportunity_reference is None or self.confidence_reference is None:
            raise RuntimeError("V21 confidence estimator is not fitted")
        margin=self._rank(self.margin_reference,np.abs(raw-self.raw_cutoff))
        opportunity=self._rank(self.opportunity_reference,self._opportunity(frame))
        confidence_raw=self.margin_weight*margin+(1.0-self.margin_weight)*opportunity
        gate=self._rank(self.confidence_reference,confidence_raw)>=self.gate_quantile
        confidence=np.where(gate,0.86+0.14*margin,0.84*margin)
        return np.where(raw>=self.raw_cutoff,0.5+0.5*confidence,0.5-0.5*confidence)

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        self.base_model=make_model(self.base_spec);self.base_model.fit(frame,y)
        raw=self.base_model.predict_proba(frame)[:,1]
        margin_raw=np.abs(raw-self.raw_cutoff);self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])
        opportunity=self._opportunity(frame,fit=True);self.opportunity_reference=np.sort(opportunity[np.isfinite(opportunity)])
        margin=self._rank(self.margin_reference,margin_raw);opportunity_rank=self._rank(self.opportunity_reference,opportunity)
        confidence_raw=self.margin_weight*margin+(1.0-self.margin_weight)*opportunity_rank
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        return self

    def predict_proba(self,frame:pd.DataFrame)->np.ndarray:
        if self.base_model is None:raise RuntimeError("V21 model is not fitted")
        probability=np.clip(self._probability(frame,self.base_model.predict_proba(frame)[:,1]),1e-6,1-1e-6)
        return np.column_stack([1-probability,probability])

class V22SupervisedConfidenceEstimator(V21SupervisedConfidenceEstimator):
    """V22 production-OOF cutoff and entry-time confidence policy."""


class V23PriorConfidenceEstimator(V21SupervisedConfidenceEstimator):
    """Frozen V22-prior direction model with V23-DEV confidence references.

    Direction probabilities for the current DEV sample are genuinely grouped
    out of fold whenever current labels are used.  The resulting OOF margins,
    plus an entry-time-only opportunity feature, are the frozen reference
    distributions later applied to the untouched seal.
    """
    def __init__(self,market:str,base_spec:dict,raw_cutoff:float,
                 opportunity_column:str,opportunity_sign:float,
                 margin_weight:float,gate_quantile:float,
                 target_repeat:int,cv_folds:int=4):
        super().__init__(market,base_spec,raw_cutoff,opportunity_column,
                         opportunity_sign,margin_weight,gate_quantile)
        self.target_repeat=int(target_repeat)
        self.cv_folds=int(cv_folds)
        self.oof_event_ids:Optional[np.ndarray]=None
        self.oof_probability:Optional[np.ndarray]=None

    def _groups(self,frame:pd.DataFrame)->pd.Series:
        if "event_id" not in frame or "event_time_utc" not in frame:
            raise RuntimeError("V23 estimator requires event_id and event_time_utc")
        if self.market=="KR":
            return frame["event_id"].astype(str).str.rsplit(":",n=1).str[-1]
        day=pd.to_datetime(frame["event_time_utc"],utc=True).dt.strftime("%Y-%m-%d")
        return frame["ticker"].astype(str)+"|"+day

    def _training_frame(self,frame:pd.DataFrame,y:Sequence[int],
                        prior_frame:pd.DataFrame,prior_y:Sequence[int]
                        )->Tuple[pd.DataFrame,np.ndarray]:
        frames=[prior_frame]+[frame]*self.target_repeat
        labels=[np.asarray(prior_y,dtype=int)]+[
            np.asarray(y,dtype=int)
        ]*self.target_repeat
        if not frames or sum(len(part) for part in frames)==0:
            raise RuntimeError("V23 prior training sample is empty")
        return pd.concat(frames,ignore_index=True,sort=False),np.concatenate(labels)

    def _fit_base(self,frame:pd.DataFrame,y:Sequence[int],
                  prior_frame:pd.DataFrame,prior_y:Sequence[int])->Any:
        train,labels=self._training_frame(frame,y,prior_frame,prior_y)
        model=make_model(self.base_spec)
        model.fit(train,labels)
        return model

    def _set_references(self,frame:pd.DataFrame,raw:np.ndarray)->None:
        margin_raw=np.abs(np.asarray(raw,dtype=float)-self.raw_cutoff)
        self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])
        opportunity=self._opportunity(frame,fit=True)
        self.opportunity_reference=np.sort(opportunity[np.isfinite(opportunity)])
        margin=self._rank(self.margin_reference,margin_raw)
        opportunity_rank=self._rank(self.opportunity_reference,opportunity)
        confidence_raw=(self.margin_weight*margin+
                        (1.0-self.margin_weight)*opportunity_rank)
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])

    def fit_with_prior(self,frame:pd.DataFrame,y:Sequence[int],
                       prior_frame:pd.DataFrame,prior_y:Sequence[int]):
        frame=frame.reset_index(drop=True)
        prior_frame=prior_frame.reset_index(drop=True)
        labels=np.asarray(y,dtype=int)
        prior_labels=np.asarray(prior_y,dtype=int)
        if len(frame)!=len(labels) or len(prior_frame)!=len(prior_labels):
            raise RuntimeError("V23 training feature/label length mismatch")
        if len(prior_frame)==0 or np.unique(prior_labels).size<2:
            raise RuntimeError("V23 opened-prior sample is insufficient")

        if self.target_repeat==0:
            self.base_model=self._fit_base(frame,labels,prior_frame,prior_labels)
            raw=self.base_model.predict_proba(frame)[:,1]
        else:
            splitter=StratifiedGroupKFold(
                n_splits=self.cv_folds,shuffle=True,random_state=SEED
            )
            raw=np.full(len(frame),np.nan,dtype=float)
            for train_index,valid_index in splitter.split(
                    frame,labels,self._groups(frame)):
                fold_model=self._fit_base(
                    frame.iloc[train_index],labels[train_index],
                    prior_frame,prior_labels,
                )
                raw[valid_index]=fold_model.predict_proba(frame.iloc[valid_index])[:,1]
            if not np.isfinite(raw).all():
                raise RuntimeError("V23 grouped OOF probabilities are incomplete")
            self.base_model=self._fit_base(frame,labels,prior_frame,prior_labels)

        self._set_references(frame,raw)
        self.oof_event_ids=frame["event_id"].astype(str).to_numpy()
        self.oof_probability=np.clip(self._probability(frame,raw),1e-6,1-1e-6)
        return self

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        raise RuntimeError("V23 estimator must be fitted with the frozen opened prior")

    def crossfit_output(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        if self.oof_event_ids is None or self.oof_probability is None:
            raise RuntimeError("V23 OOF output is unavailable")
        event_ids=frame["event_id"].astype(str).to_numpy()
        if not np.array_equal(event_ids,self.oof_event_ids):
            lookup={event_id:index for index,event_id in enumerate(self.oof_event_ids)}
            try:
                probability=np.asarray([
                    self.oof_probability[lookup[event_id]] for event_id in event_ids
                ])
            except KeyError as exc:
                raise RuntimeError("V23 OOF event mismatch") from exc
        else:
            probability=self.oof_probability.copy()
        return probability,np.ones(len(frame),dtype=bool)


class V24PriorConfidenceEstimator(V23PriorConfidenceEstimator):
    """V24 opened-prior direction and frozen OOF confidence policy."""


class V25StableRankEstimator(V20StableRankEstimator):
    """V25 cross-version-audited, label-independent monotone rank policy."""

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        del y
        columns={column for column,_,_ in self.components}|{self.opportunity_column}
        self.references={}
        raw_values={}
        for column in sorted(columns):
            raw=pd.to_numeric(frame[column],errors="coerce").to_numpy(float)
            raw_values[column]=raw
            finite=raw[np.isfinite(raw)]
            self.medians[column]=float(np.median(finite)) if len(finite) else 0.0
            # Preserve the exact pre-freeze audit convention: missing feature
            # rows are median-imputed when scored but do not add artificial
            # mass to the empirical reference distribution.
            self.references[column]=np.sort(finite)
        direction=self.direction_score(frame)
        margin_raw=np.abs(direction-self.direction_cutoff)
        self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])
        opportunity_raw=self.opportunity_sign*raw_values[self.opportunity_column]
        self.opportunity_reference=np.sort(
            opportunity_raw[np.isfinite(opportunity_raw)]
        )
        margin=self._rank(self.margin_reference,margin_raw)
        opportunity=self.opportunity_sign*self._values(frame,self.opportunity_column)
        opportunity_rank=self._rank(self.opportunity_reference,opportunity)
        confidence_raw=(
            self.margin_weight*margin+(1.0-self.margin_weight)*opportunity_rank
        )
        self.confidence_reference=np.sort(confidence_raw[np.isfinite(confidence_raw)])
        self.oof_event_ids=frame["event_id"].astype(str).to_numpy()
        self.oof_probability=self._probability(frame,direction)
        return self


class V26StableRankEstimator(V25StableRankEstimator):
    """V26 opened-audit and internal-split stable US rank policy."""


class V27StableRankEstimator(V25StableRankEstimator):
    """V27 rank direction/confidence with an optional entry-only eligibility gate."""

    def __init__(self,market:str,direction_cutoff:float,components:Sequence[Sequence[Any]],
                 opportunity_column:str,opportunity_sign:float,
                 margin_weight:float,gate_quantile:float,
                 eligibility:Optional[dict]=None):
        super().__init__(market,direction_cutoff,components,opportunity_column,
                         opportunity_sign,margin_weight,gate_quantile)
        self.eligibility_spec=dict(eligibility) if eligibility else None
        self.eligibility_median:float=0.0
        self.eligibility_margin_reference:Optional[np.ndarray]=None
        self.eligibility_opportunity_reference:Optional[np.ndarray]=None
        self.eligibility_confidence_reference:Optional[np.ndarray]=None

    def _eligibility_opportunity(self,frame:pd.DataFrame,fit:bool=False)->np.ndarray:
        if self.eligibility_spec is None:
            return np.zeros(len(frame),dtype=float)
        raw=pd.to_numeric(
            frame[self.eligibility_spec["column"]],errors="coerce"
        ).to_numpy(float)
        signed=float(self.eligibility_spec["sign"])*raw
        finite=signed[np.isfinite(signed)]
        if fit:
            self.eligibility_median=float(np.median(finite)) if len(finite) else 0.0
        return np.nan_to_num(
            signed,nan=self.eligibility_median,
            posinf=self.eligibility_median,neginf=self.eligibility_median,
        )

    def fit(self,frame:pd.DataFrame,y:Sequence[int]):
        super().fit(frame,y)
        # Keep DEV crossfit and frozen transfer on the same valid-probability
        # contract, including empirical-rank endpoint observations.
        if self.oof_probability is not None:
            self.oof_probability=np.clip(self.oof_probability,1e-6,1.0-1e-6)
        if self.eligibility_spec is None:
            return self
        direction=self.direction_score(frame)
        margin_raw=np.abs(direction-self.direction_cutoff)
        self.eligibility_margin_reference=np.sort(
            margin_raw[np.isfinite(margin_raw)]
        )
        raw=pd.to_numeric(
            frame[self.eligibility_spec["column"]],errors="coerce"
        ).to_numpy(float)
        signed=float(self.eligibility_spec["sign"])*raw
        self.eligibility_opportunity_reference=np.sort(
            signed[np.isfinite(signed)]
        )
        opportunity=self._eligibility_opportunity(frame,fit=True)
        margin=self._rank(self.eligibility_margin_reference,margin_raw)
        opportunity_rank=self._rank(
            self.eligibility_opportunity_reference,opportunity
        )
        weight=float(self.eligibility_spec["margin_weight"])
        confidence=weight*margin+(1.0-weight)*opportunity_rank
        self.eligibility_confidence_reference=np.sort(
            confidence[np.isfinite(confidence)]
        )
        return self

    def eligible(self,frame:pd.DataFrame)->np.ndarray:
        if self.eligibility_spec is None:
            return np.ones(len(frame),dtype=bool)
        if (self.eligibility_margin_reference is None or
                self.eligibility_opportunity_reference is None or
                self.eligibility_confidence_reference is None):
            raise RuntimeError("V27 eligibility policy is not fitted")
        direction=self.direction_score(frame)
        margin=self._rank(
            self.eligibility_margin_reference,
            np.abs(direction-self.direction_cutoff),
        )
        opportunity=self._rank(
            self.eligibility_opportunity_reference,
            self._eligibility_opportunity(frame),
        )
        weight=float(self.eligibility_spec["margin_weight"])
        confidence=weight*margin+(1.0-weight)*opportunity
        return self._rank(
            self.eligibility_confidence_reference,confidence
        )>=float(self.eligibility_spec["quantile"])

    def crossfit_output(self,frame:pd.DataFrame)->Tuple[np.ndarray,np.ndarray]:
        probability,_=super().crossfit_output(frame)
        return probability,self.eligible(frame)


class V28StableRankEstimator(V27StableRankEstimator):
    """V28 cross-audited entry-feature rank direction and confidence policy."""


class V29StableRankEstimator(V28StableRankEstimator):
    """V29 cross-audited entry-feature rank direction and confidence policy."""


class V30StableRankEstimator(V29StableRankEstimator):
    """V30 cross-audited entry-feature rank direction and confidence policy."""


class V31StableRankEstimator(V30StableRankEstimator):
    """V31 stable rank policy using the exact pre-freeze midrank convention."""

    @staticmethod
    def _rank(reference:np.ndarray,values:np.ndarray)->np.ndarray:
        reference=np.sort(np.asarray(reference,dtype=float)[np.isfinite(reference)])
        values=np.asarray(values,dtype=float)
        if not len(reference):
            return np.full(len(values),0.5,dtype=float)
        left=np.searchsorted(reference,values,side="left")
        right=np.searchsorted(reference,values,side="right")
        return (left+right+1.0)/(2.0*len(reference))


class V32StableRankEstimator(V31StableRankEstimator):
    """V31-DEV-to-SEAL transfer policy frozen before the pure V32 seal."""


class V33StableRankEstimator(V32StableRankEstimator):
    """Opened-V32-audited transfer policy frozen before the pure V33 seal."""


class V34StableRankEstimator(V33StableRankEstimator):
    """Opened-V33-audited transfer policy frozen before the pure V34 seal."""


class V35StableRankEstimator(V34StableRankEstimator):
    """Opened-V34-audited transfer policy frozen before the pure V35 seal."""


class V26PriorConfidenceEstimator(V23PriorConfidenceEstimator):
    """V26 deterministic prior-augmented KR direction and confidence policy."""

    def _set_references(self,frame:pd.DataFrame,raw:np.ndarray)->None:
        """Match the frozen audit's missing-value empirical-rank convention."""
        margin_raw=np.abs(np.asarray(raw,dtype=float)-self.raw_cutoff)
        self.margin_reference=np.sort(margin_raw[np.isfinite(margin_raw)])

        values=pd.to_numeric(
            frame[self.opportunity_column],errors="coerce"
        ).to_numpy(float)
        finite=values[np.isfinite(values)]
        self.opportunity_median=float(np.median(finite)) if len(finite) else 0.0
        opportunity_raw=self.opportunity_sign*values
        self.opportunity_reference=np.sort(
            opportunity_raw[np.isfinite(opportunity_raw)]
        )

        margin=self._rank(self.margin_reference,margin_raw)
        opportunity_rank=self._rank(
            self.opportunity_reference,self._opportunity(frame)
        )
        confidence_raw=(self.margin_weight*margin+
                        (1.0-self.margin_weight)*opportunity_rank)
        self.confidence_reference=np.sort(
            confidence_raw[np.isfinite(confidence_raw)]
        )


def make_numeric_model(spec:dict)->Pipeline:
    cat_cols=list(spec.get("cat_cols",["event_type"]))
    dense=spec["kind"]=="cat"
    num=Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=True)),
    ])
    pre=ColumnTransformer([
        ("num",num,NUMERIC_COLS),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=not dense),cat_cols),
    ],sparse_threshold=0.0 if dense else 0.2)
    if spec["kind"]=="lgb":
        clf=LGBMClassifier(
            n_estimators=int(spec["n_estimators"]),
            num_leaves=int(spec["num_leaves"]),learning_rate=0.035,
            min_child_samples=25,subsample=0.8,colsample_bytree=0.8,
            reg_lambda=3.0,verbosity=-1,random_state=SEED,
            class_weight="balanced",n_jobs=runtime_limits.THREAD_COUNT,device_type="cpu",
        )
    else:
        clf=CatBoostClassifier(
            iterations=int(spec["iterations"]),depth=int(spec["depth"]),
            learning_rate=0.035,l2_leaf_reg=5.0,verbose=False,
            random_seed=SEED,auto_class_weights="Balanced",
            thread_count=runtime_limits.THREAD_COUNT,
            task_type=runtime_limits.CATBOOST_TASK_TYPE,
            devices=runtime_limits.CATBOOST_DEVICES,
            allow_writing_files=False,
        )
    return Pipeline([("pre",pre),("clf",clf)])

def make_model(spec:dict):
    if spec["kind"]=="v35_stable_rank":
        return V35StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v34_stable_rank":
        return V34StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v33_stable_rank":
        return V33StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v32_stable_rank":
        return V32StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v31_stable_rank":
        return V31StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v30_stable_rank":
        return V30StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v29_stable_rank":
        return V29StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v28_stable_rank":
        return V28StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v27_stable_rank":
        return V27StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec.get("eligibility"),
        )
    if spec["kind"]=="v26_stable_rank":
        return V26StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v26_prior_confidence":
        return V26PriorConfidenceEstimator(
            spec["market"],spec["base_spec"],spec["raw_cutoff"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec["target_repeat"],spec.get("cv_folds",4),
        )
    if spec["kind"]=="v25_stable_rank":
        return V25StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v24_prior_confidence":
        return V24PriorConfidenceEstimator(
            spec["market"],spec["base_spec"],spec["raw_cutoff"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec["target_repeat"],spec.get("cv_folds",4),
        )
    if spec["kind"]=="v23_prior_confidence":
        return V23PriorConfidenceEstimator(
            spec["market"],spec["base_spec"],spec["raw_cutoff"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
            spec["target_repeat"],spec.get("cv_folds",4),
        )
    if spec["kind"]=="v22_supervised_confidence":
        return V22SupervisedConfidenceEstimator(
            spec["market"],spec["base_spec"],spec["raw_cutoff"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v21_supervised_confidence":
        return V21SupervisedConfidenceEstimator(
            spec["market"],spec["base_spec"],spec["raw_cutoff"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v21_stable_rank":
        return V21StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v20_stable_rank":
        return V20StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v19_stable_rank":
        return V19StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v18_prior_transfer":
        return V18PriorTransferEstimator(
            spec["market"],spec["base_spec"],spec["raw_cutoff"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v17_stable_rank":
        return V17StableRankEstimator(
            spec["market"],spec["direction_cutoff"],spec["components"],
            spec["opportunity_column"],spec["opportunity_sign"],
            spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v16_stable_rank":
        raise RuntimeError("V16 estimator is available only in the archived V16 source")
    if spec["kind"]=="v15_repeated_opportunity":
        return V15RepeatedOpportunityEstimator(
            spec["market"],spec["direction_cutoff"],spec["opportunity_column"],
            spec["opportunity_sign"],spec["margin_weight"],spec["gate_quantile"],
        )
    if spec["kind"]=="v14_repeated_opportunity":
        return V14RepeatedOpportunityEstimator(
            spec["market"],spec["direction_cutoff"],spec["opportunity_column"],
            spec["opportunity_sign"],spec.get("margin_weight",0.25),
        )
    if spec["kind"]=="v13_repeated_rank":
        return RepeatedEntryRankEstimator(spec["market"])
    if spec["kind"]=="kr_rank_blend":
        return KoreanRankBlendEstimator(spec.get("text_C",0.01),spec.get("probability_scale",8.0))
    if spec["kind"]=="us_context_rule":
        return USContextRuleEstimator()
    if spec["kind"]=="mean_reversion":
        return MeanReversionEstimator(
            spec["weights"],spec["scale_quantile"],spec["scale_confidence"],
            spec.get("columns",("pre_ret_2","pre_ret_5","pre_ret_15")),
        )
    if spec["kind"]=="native_cat":
        return NativeCatEstimator(
            int(spec["iterations"]),int(spec["depth"]),spec["cat_cols"]
        )
    if spec["kind"] in {"lgb","cat"}:
        return make_numeric_model(spec)
    if spec["kind"]=="blend":
        return ProbabilityBlendEstimator(spec["components"],spec["weights"])
    num=Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",StandardScaler(with_mean=False)),
    ])
    pre=ColumnTransformer([
        ("text",TfidfVectorizer(
            lowercase=True,
            ngram_range=(1,2),
            min_df=2,
            max_df=0.995,
            max_features=30000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),TEXT_COL),
        ("num",num,NUMERIC_COLS),
        ("cat",OneHotEncoder(handle_unknown="ignore"),CAT_COLS),
    ],sparse_threshold=0.2)

    if spec["kind"]=="logit":
        clf=LogisticRegression(
            C=spec["C"],class_weight=spec["class_weight"],
            max_iter=1800,solver="liblinear",random_state=SEED
        )
    else:
        clf=SGDClassifier(
            loss="log_loss",alpha=spec["alpha"],class_weight=spec["class_weight"],
            max_iter=2500,tol=1e-4,random_state=SEED
        )
    return Pipeline([("pre",pre),("clf",clf)])

def split_dev_seal(df:pd.DataFrame,cfg:Config):
    x=df.sort_values("event_time_utc").copy()
    x["event_time_utc"]=pd.to_datetime(x["event_time_utc"],utc=True)

    if cfg.fixed_seal_sources:
        dev_parts=[]; seal_parts=[]; starts={}
        for market,source_name in cfg.fixed_seal_sources:
            g=x[x.market.eq(market)]
            part_dev=g[~g.source.eq(source_name)].copy()
            part_seal=g[g.source.eq(source_name)].copy()
            if part_dev.empty or part_seal.empty:
                raise RuntimeError(f"fixed source split empty: {market}/{source_name}")
            dev_parts.append(part_dev); seal_parts.append(part_seal)
            starts[market]=part_seal.event_time_utc.min()
        dev=pd.concat(dev_parts).sort_values("event_time_utc")
        seal=pd.concat(seal_parts).sort_values("event_time_utc")
        if set(dev.source).intersection(set(seal.source)):
            raise RuntimeError("DEV/SEAL source overlap")
        if (dev["event_time_utc"].max()-dev["event_time_utc"].min()).days<cfg.dev_min_days:
            raise RuntimeError("DEV period is too short")
        return dev,seal,starts

    if cfg.fixed_seal_windows:
        dev_parts=[]; seal_parts=[]; starts={}
        for market,raw_start,raw_end in cfg.fixed_seal_windows:
            start=pd.Timestamp(raw_start)
            end=pd.Timestamp(raw_end)
            if start.tzinfo is None: start=start.tz_localize("UTC")
            else: start=start.tz_convert("UTC")
            if end.tzinfo is None: end=end.tz_localize("UTC")
            else: end=end.tz_convert("UTC")
            g=x[x.market.eq(market)]
            part_dev=g[g.event_time_utc<start].copy()
            part_seal=g[(g.event_time_utc>=start)&(g.event_time_utc<end)].copy()
            if part_dev.empty or part_seal.empty:
                raise RuntimeError(f"fixed split empty: {market}")
            dev_parts.append(part_dev); seal_parts.append(part_seal); starts[market]=start
        dev=pd.concat(dev_parts).sort_values("event_time_utc")
        seal=pd.concat(seal_parts).sort_values("event_time_utc")
        if (dev["event_time_utc"].max()-dev["event_time_utc"].min()).days<cfg.dev_min_days:
            raise RuntimeError("DEV 기간이 너무 짧습니다.")
        return dev,seal,starts

    groups = list(x.groupby("market", sort=True)) if cfg.market_stratified_split else [("ALL", x)]
    dev_parts=[]; seal_parts=[]; starts={}
    for market,g in groups:
        g=g.sort_values("event_time_utc")
        n=len(g)
        cut=min(max(int(n*(1-cfg.seal_fraction)),1),n-1)
        ss=g.iloc[cut]["event_time_utc"]
        # ensure at least seal_min_days if possible
        candidate=g["event_time_utc"].max()-pd.Timedelta(days=cfg.seal_min_days)
        if candidate < ss:
            ss=candidate
        part_dev=g[g["event_time_utc"]<ss].copy()
        part_seal=g[g["event_time_utc"]>=ss].copy()
        if part_dev.empty or part_seal.empty:
            raise RuntimeError(f"split empty: {market}")
        starts[market]=ss
        dev_parts.append(part_dev); seal_parts.append(part_seal)
    dev=pd.concat(dev_parts).sort_values("event_time_utc")
    seal=pd.concat(seal_parts).sort_values("event_time_utc")
    if dev.empty or seal.empty:
        raise RuntimeError("split empty")
    if (dev["event_time_utc"].max()-dev["event_time_utc"].min()).days<cfg.dev_min_days:
        raise RuntimeError("DEV 기간이 너무 짧습니다.")
    return dev,seal,starts

def cv_folds(dev:pd.DataFrame,nfolds:int):
    qs=np.linspace(0.45,0.90,nfolds)
    folds=[]
    for q in qs:
        train_indices=[]; valid_indices=[]
        for _,g in dev.groupby("market",sort=True):
            cut=pd.Timestamp(g["event_time_utc"].quantile(q))
            nxt=pd.Timestamp(g["event_time_utc"].quantile(min(q+0.10,0.995)))
            train_indices.extend(g.index[g["event_time_utc"]<cut].tolist())
            valid_indices.extend(g.index[(g["event_time_utc"]>=cut)&
                                         (g["event_time_utc"]<nxt)].tolist())
        tr=dev.loc[train_indices]
        va=dev.loc[valid_indices]
        if len(tr)>=150 and len(va)>=30:
            folds.append((tr.index,va.index))
    if len(folds)<2:
        raise RuntimeError("CV fold insufficient")
    return folds

def evaluate(df:pd.DataFrame,thr:float,cost:float)->dict:
    y=df["y"].astype(int).to_numpy()
    p=df["prob"].astype(float).to_numpy()
    weight=(
        pd.to_numeric(df["eval_weight"],errors="coerce").fillna(0).to_numpy(float)
        if "eval_weight" in df.columns else np.ones(len(df),dtype=float)
    )
    if not np.isfinite(weight).all() or weight.sum()<=0:
        raise ValueError("evaluation weights must be finite with positive total")
    pred=(p>=0.5).astype(int)
    acc=float(accuracy_score(y,pred,sample_weight=weight))
    bal=float(balanced_accuracy_score(y,pred,sample_weight=weight))
    auc=safe_auc(y,p,weight)
    base=float(np.average(y,weights=weight))
    naive=max(base,1-base)
    conf=np.maximum(p,1-p)
    m=conf>=thr
    hc_acc=float(np.average((pred[m]==y[m]).astype(float),weights=weight[m])) if m.sum() else float("nan")
    hc_cov=float(weight[m].sum()/weight.sum())
    direction=np.where(pred==1,1.0,-1.0)
    signed=direction*df["fwd_ret_30m"].astype(float).to_numpy()
    net=signed-cost
    strategy_gross=signed[m]
    strategy_net=strategy_gross-cost

    by_market={}
    for market,g in df.groupby("market"):
        yy=g["y"].astype(int).to_numpy()
        pp=g["prob"].astype(float).to_numpy()
        ww=(
            pd.to_numeric(g["eval_weight"],errors="coerce").fillna(0).to_numpy(float)
            if "eval_weight" in g.columns else np.ones(len(g),dtype=float)
        )
        pr=(pp>=0.5).astype(int)
        by_market[market]={
            "n":int(len(g)),
            "accuracy":float(accuracy_score(yy,pr,sample_weight=ww)),
            "balanced_accuracy":float(balanced_accuracy_score(yy,pr,sample_weight=ww)) if len(np.unique(yy))>1 else float("nan"),
            "auc":safe_auc(yy,pp,ww),
            "mean_signed_net":float(np.average(
                np.where(pr==1,1.0,-1.0)*g["fwd_ret_30m"].to_numpy()-cost,
                weights=ww,
            )),
        }

    return {
        "n":int(len(df)),
        "accuracy":acc,
        "balanced_accuracy":bal,
        "auc":auc,
        "up_rate":base,
        "naive_accuracy":naive,
        "edge_vs_naive":acc-naive,
        "highconf_threshold":float(thr),
        "highconf_n":int(m.sum()),
        "highconf_coverage":hc_cov,
        "highconf_accuracy":hc_acc,
        "mean_signed_gross":float(np.average(signed,weights=weight)),
        "mean_signed_net":float(np.average(net,weights=weight)),
        "strategy_mean_signed_gross":float(np.average(strategy_gross,weights=weight[m])) if m.sum() else float("nan"),
        "strategy_mean_signed_net":float(np.average(strategy_net,weights=weight[m])) if m.sum() else float("nan"),
        "median_signed_net":float(np.median(net)),
        "by_market":by_market,
    }

def bootstrap_by_day(df:pd.DataFrame,thr:float,cost:float,nboot:int=600)->dict:
    x=df.copy().reset_index(drop=True)
    x["day"]=pd.to_datetime(x["event_time_utc"],utc=True).dt.date
    # Build day -> row-position arrays once.  The prior implementation scanned
    # the full frame once for every sampled day inside every replicate; this is
    # mathematically identical but avoids billions of redundant comparisons on
    # sparse, multi-year event panels.
    positions_by_day={
        day:np.asarray(positions,dtype=int)
        for day,positions in x.groupby("day",sort=True).indices.items()
    }
    days=np.array(sorted(positions_by_day),dtype=object)
    rng=np.random.default_rng(SEED+99)
    bals=[]; nets=[]
    for _ in range(nboot):
        sample=rng.choice(days,size=len(days),replace=True)
        positions=np.concatenate([positions_by_day[d] for d in sample])
        b=x.iloc[positions]
        if b["y"].nunique()<2:
            continue
        m=evaluate(b,thr,cost)
        bals.append(m["balanced_accuracy"])
        nets.append(m["mean_signed_net"])
    return {
        "n_boot":len(bals),
        "balanced_accuracy_lower95":float(np.quantile(bals,0.025)) if bals else float("nan"),
        "balanced_accuracy_upper95":float(np.quantile(bals,0.975)) if bals else float("nan"),
        "mean_signed_net_lower95":float(np.quantile(nets,0.025)) if nets else float("nan"),
        "mean_signed_net_upper95":float(np.quantile(nets,0.975)) if nets else float("nan"),
    }

def choose_threshold(oof:pd.DataFrame,cfg:Config):
    if cfg.fixed_highconf_threshold is not None:
        threshold=float(cfg.fixed_highconf_threshold)
        return threshold,evaluate(oof,threshold,cfg.round_trip_cost)
    best=None
    thresholds=sorted(set([0.54,0.56,0.58,0.60,0.62,0.65,0.68,0.72]+[
        round(float(t),3) for t in np.arange(0.75,0.976,0.025)
    ]))
    for t in thresholds:
        m=evaluate(oof,t,cfg.round_trip_cost)
        if m["highconf_coverage"]<cfg.cert_min_highconf_coverage or not np.isfinite(m["highconf_accuracy"]):
            continue
        clears_strategy=(
            m["highconf_accuracy"]>=cfg.cert_min_highconf_acc and
            m["strategy_mean_signed_net"]>=cfg.cert_min_mean_signed_net
        )
        score=(
            2.0*m["balanced_accuracy"]+
            1.5*(m["auc"] if np.isfinite(m["auc"]) else 0.5)+
            3.0*m["highconf_accuracy"]+
            0.4*m["highconf_coverage"]+
            30.0*m["strategy_mean_signed_net"]+
            (2.0 if clears_strategy else 0.0)
        )
        if best is None or score>best[0]:
            best=(score,t,m)
    return (best[1],best[2]) if best else (0.60,evaluate(oof,0.60,cfg.round_trip_cost))

def cv_spec(dev:pd.DataFrame,spec:dict,cfg:Config):
    preds=[]
    folds=cv_folds(dev,cfg.cv_folds)
    for fi,(tri,vai) in enumerate(folds,1):
        tr=dev.loc[tri].copy(); va=dev.loc[vai].copy()
        model=make_model(spec)
        model.fit(model_frame(tr),tr["y"].astype(int))
        p=model.predict_proba(model_frame(va))[:,1]
        z=va[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
        z["prob"]=p
        z["fold"]=fi
        preds.append(z)
    oof=pd.concat(preds,ignore_index=True)
    thr,met=choose_threshold(oof,cfg)
    # DEV only selection score
    score=(
        2.0*met["balanced_accuracy"]+
        1.5*(met["auc"] if np.isfinite(met["auc"]) else .5)+
        2.0*met["highconf_accuracy"]+
        0.8*met["highconf_coverage"]+
        18.0*met["strategy_mean_signed_net"]
    )
    return {"spec":spec,"score":float(score),"threshold":thr,"metrics":met}

def cv_market_predictions(dev:pd.DataFrame,market:str,spec:dict,cfg:Config)->pd.DataFrame:
    """Article/day-grouped OOF on the predeclared current DEV source."""
    validation_sources=[
        source_name for source_market,source_name in cfg.dev_validation_sources
        if source_market==market
    ]
    source=dev[dev.market.eq(market)&dev.source.isin(validation_sources)].copy()
    source=source.sort_values("event_time_utc").reset_index(drop=True)
    if source.empty or source.y.nunique()<2:
        raise RuntimeError(f"market validation source insufficient: {market} {validation_sources}")
    cutoff=float(dict(cfg.decision_cutoff_by_market).get(market,0.5))
    invert=bool(dict(cfg.invert_probability_by_market).get(market,False))
    preds=[]

    if spec.get("kind") in {"v32_stable_rank","v33_stable_rank","v34_stable_rank",
                             "v35_stable_rank"}:
        reference_sources=set(spec.get("reference_sources",[]))
        reference=(dev[dev.market.eq(market)&dev.source.isin(reference_sources)&dev.y.notna()]
                   .sort_values("event_time_utc").reset_index(drop=True))
        if reference.empty or reference.y.nunique()<2:
            raise RuntimeError(
                f"stable-rank transfer reference insufficient: {market} {sorted(reference_sources)}"
            )
        model=make_model(spec)
        model.fit(model_frame(reference),reference["y"].astype(int))
        raw=model.predict_proba(model_frame(source))[:,1]
        if invert:raw=1.0-raw
        eligible=model.eligible(model_frame(source))
        z=source[[
            "event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"
        ]].copy()
        z["prob"]=raw
        z=z.loc[eligible].reset_index(drop=True)
        if z.empty or z.y.nunique()<2:
            raise RuntimeError(f"stable-rank transfer output insufficient: {market}")
        target=float(dict(cfg.dev_market_weight_targets).get(market,len(z)))
        z["eval_weight"]=target/len(z)
        return z

    if spec.get("kind") in {
        "v13_repeated_rank","v14_repeated_opportunity","v15_repeated_opportunity",
        "v16_stable_rank","v17_stable_rank","v18_prior_transfer","v19_stable_rank",
        "v20_stable_rank",
        "v21_stable_rank","v25_stable_rank","v26_stable_rank","v27_stable_rank",
        "v28_stable_rank","v29_stable_rank","v30_stable_rank","v31_stable_rank",
        "v23_prior_confidence","v24_prior_confidence","v26_prior_confidence",
    }:
        model=make_model(spec)
        features=model_frame(source)
        if spec.get("kind") in {"v23_prior_confidence","v24_prior_confidence","v26_prior_confidence"}:
            augmentation_sources=set(spec.get("augmentation_sources",[]))
            prior=dev[
                dev.market.eq(market)&dev.source.isin(augmentation_sources)&dev.y.notna()
            ].copy().sort_values("event_time_utc").reset_index(drop=True)
            model.fit_with_prior(
                features,source["y"].astype(int),
                model_frame(prior),prior["y"].astype(int),
            )
        elif spec.get("kind")=="v18_prior_transfer":
            prior=dev[
                dev.market.eq(market)&~dev.source.isin(validation_sources)&dev.y.notna()
            ].copy()
            model.fit_with_prior(
                features,source["y"].astype(int),model_frame(prior),prior["y"].astype(int)
            )
        elif spec.get("kind")=="v15_repeated_opportunity":
            augmentation_sources=set(spec.get("augmentation_sources",[]))
            prior=dev[
                dev.market.eq(market)&dev.source.isin(augmentation_sources)&dev.y.notna()
            ].copy()
            prior_features=model_frame(prior) if not prior.empty else None
            prior_labels=prior["y"].astype(int) if not prior.empty else None
            model.fit_with_prior(
                features,source["y"].astype(int),prior_features,prior_labels
            )
        else:
            model.fit(features,source["y"].astype(int))
        raw,eligible=model.crossfit_output(features)
        if spec.get("kind")=="v13_repeated_rank":
            if invert:
                raw=1.0-raw
            raw=shift_probability(raw,cutoff)
        elif invert:
            raw=1.0-raw
        z=source[["event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"]].copy()
        z["prob"]=raw
        z=z.loc[eligible].reset_index(drop=True)
        if z.empty or z.y.nunique()<2:
            raise RuntimeError(f"eligible OOF insufficient: {market}")
        target=float(dict(cfg.dev_market_weight_targets).get(market,len(z)))
        z["eval_weight"]=target/len(z)
        return z

    day=pd.to_datetime(source["event_time_utc"],utc=True).dt.strftime("%Y-%m-%d")
    if market=="KR":
        # One Naver article can tag several tickers.  Keep every copy in one
        # fold so article text/labels cannot leak across train and validation.
        groups=source["event_id"].astype(str).str.rsplit(":",n=1).str[-1]
    elif market=="US":
        groups=source["ticker"].astype(str)+"|"+day
    else:
        raise RuntimeError(f"unsupported target-matched market: {market}")
    splitter=StratifiedGroupKFold(n_splits=cfg.cv_folds,shuffle=True,random_state=SEED)
    split_indices=list(splitter.split(source,source["y"].astype(int),groups))

    for fi,(train_index,valid_index) in enumerate(split_indices,1):
        tr=source.iloc[train_index].copy()
        va=source.iloc[valid_index].copy()
        if tr.empty or va.empty or tr.y.nunique()<2:
            raise RuntimeError(f"market CV fold insufficient: {market} fold={fi}")
        model=make_model(spec)
        model.fit(model_frame(tr),tr["y"].astype(int))
        raw=model.predict_proba(model_frame(va))[:,1]
        if invert:
            raw=1.0-raw
        z=va[["event_id","event_time_utc","market","source","ticker","y","fwd_ret_30m"]].copy()
        z["prob"]=shift_probability(raw,cutoff)
        z["fold"]=fi
        preds.append(z)
    if not preds:
        raise RuntimeError(f"market CV insufficient: {market} {spec}")
    result=pd.concat(preds,ignore_index=True)
    target=float(dict(cfg.dev_market_weight_targets).get(market,len(result)))
    result["eval_weight"]=target/len(result)
    return result

def predict_proba_frozen(model:Any,frame:pd.DataFrame)->np.ndarray:
    """Predict with either a legacy single pipeline or V5 per-market pipelines."""
    features=model_frame(frame)
    if not isinstance(model,dict) or "by_market" not in model:
        return model.predict_proba(features)[:,1]
    probabilities=np.full(len(frame),np.nan,dtype=float)
    for market,pipeline in model["by_market"].items():
        positions=np.flatnonzero(frame["market"].astype(str).to_numpy()==market)
        if len(positions):
            raw=pipeline.predict_proba(features.iloc[positions])[:,1]
            if bool(model.get("invert_probability_by_market",{}).get(market,False)):
                raw=1.0-raw
            cutoff=float(model.get("decision_cutoff_by_market",{}).get(market,0.5))
            probabilities[positions]=shift_probability(raw,cutoff)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("frozen model has no pipeline for one or more markets")
    return probabilities

def predict_eligible_frozen(model:Any,frame:pd.DataFrame)->np.ndarray:
    """Apply a frozen, outcome-independent event eligibility policy."""
    if not isinstance(model,dict) or "by_market" not in model:
        return np.ones(len(frame),dtype=bool)
    features=model_frame(frame)
    eligible=np.zeros(len(frame),dtype=bool)
    for market,pipeline in model["by_market"].items():
        positions=np.flatnonzero(frame["market"].astype(str).to_numpy()==market)
        if not len(positions):
            continue
        if hasattr(pipeline,"eligible"):
            eligible[positions]=np.asarray(
                pipeline.eligible(features.iloc[positions]),dtype=bool
            )
        else:
            eligible[positions]=True
    return eligible

def dev_selection_score(metrics:dict,cfg:Config)->Tuple[int,float]:
    by=metrics["by_market"]
    required=[
        metrics["balanced_accuracy"]>=cfg.cert_min_bal_acc,
        metrics["auc"]>=cfg.cert_min_auc,
        metrics["edge_vs_naive"]>=cfg.cert_min_edge_vs_naive,
        metrics["highconf_coverage"]>=cfg.cert_min_highconf_coverage,
        metrics["highconf_accuracy"]>=cfg.cert_min_highconf_acc,
        metrics["strategy_mean_signed_net"]>=cfg.cert_min_mean_signed_net,
        by.get("US",{}).get("balanced_accuracy",-np.inf)>=cfg.cert_min_market_bal_acc,
        by.get("KR",{}).get("balanced_accuracy",-np.inf)>=cfg.cert_min_market_bal_acc,
    ]
    score=(
        4.0*metrics["balanced_accuracy"]+
        2.0*metrics["auc"]+
        2.0*metrics["highconf_accuracy"]+
        0.2*min(metrics["highconf_coverage"],0.50)+
        25.0*metrics["strategy_mean_signed_net"]
    )
    return sum(bool(x) for x in required),float(score)

def search_and_freeze(df:pd.DataFrame,cfg:Config):
    if FROZEN_SPEC.exists() or FROZEN_MODEL.exists():
        raise RuntimeError("이미 frozen artifact가 있습니다. 같은 V3에서 재검색 금지.")
    dev,seal,starts=split_dev_seal(df,cfg)
    print(f"[DEV] {dev['event_time_utc'].min()} -> {dev['event_time_utc'].max()} n={len(dev)}")
    for market,ss in starts.items():
        part=seal[seal.market.eq(market)] if market!="ALL" else seal
        print(f"[SEALED untouched {market}] {ss} -> {part['event_time_utc'].max()} n={len(part)}")
    markets=sorted(dev.market.unique())
    specs_by_market={market:model_specs(market) for market in markets}
    oof={}
    for market in markets:
        specs=specs_by_market[market]
        for i,spec in enumerate(specs,1):
            oof[(market,i-1)]=cv_market_predictions(dev,market,spec,cfg)
            print(f"[CV {market}] {i}/{len(specs)} {spec}")

    results=[]
    if markets==["KR","US"]:
        kr_specs=specs_by_market["KR"]
        us_specs=specs_by_market["US"]
        for kr_i in range(len(kr_specs)):
            for us_i in range(len(us_specs)):
                combined=pd.concat([oof[("KR",kr_i)],oof[("US",us_i)]],ignore_index=True)
                threshold,metrics=choose_threshold(combined,cfg)
                pass_count,score=dev_selection_score(metrics,cfg)
                results.append({
                    "spec_by_market":{"KR":kr_specs[kr_i],"US":us_specs[us_i]},
                    "pass_count":pass_count,"score":score,
                    "threshold":threshold,"metrics":metrics,
                })
    else:
        market=markets[0]
        specs=specs_by_market[market]
        for i,spec in enumerate(specs):
            threshold,metrics=choose_threshold(oof[(market,i)],cfg)
            pass_count,score=dev_selection_score(metrics,cfg)
            results.append({
                "spec_by_market":{market:spec},"pass_count":pass_count,
                "score":score,"threshold":threshold,"metrics":metrics,
            })
    results.sort(key=lambda x:(x["pass_count"],x["score"]),reverse=True)
    win=results[0]
    print("[DEV WINNER] "+json.dumps(win,ensure_ascii=False,indent=2))
    pipelines={}
    final_sources={
        market:{source for source_market,source in cfg.final_training_sources
                if source_market==market}
        for market in markets
    }
    for market,spec in win["spec_by_market"].items():
        part=dev[dev.market.eq(market)&dev.source.isin(final_sources[market])]
        if part.empty or part.y.nunique()<2:
            raise RuntimeError(f"final training source insufficient: {market} {final_sources[market]}")
        pipeline=make_model(spec)
        features=model_frame(part)
        if spec.get("kind") in {"v23_prior_confidence","v24_prior_confidence","v26_prior_confidence"}:
            augmentation_sources=set(spec.get("augmentation_sources",[]))
            prior=dev[
                dev.market.eq(market)&dev.source.isin(augmentation_sources)&dev.y.notna()
            ].copy().sort_values("event_time_utc").reset_index(drop=True)
            pipeline.fit_with_prior(
                features,part["y"].astype(int),
                model_frame(prior),prior["y"].astype(int),
            )
        elif spec.get("kind")=="v18_prior_transfer":
            validation_sources={
                source for source_market,source in cfg.dev_validation_sources
                if source_market==market
            }
            prior=dev[
                dev.market.eq(market)&~dev.source.isin(validation_sources)&dev.y.notna()
            ].copy()
            pipeline.fit_with_prior(
                features,part["y"].astype(int),model_frame(prior),prior["y"].astype(int)
            )
        elif spec.get("kind")=="v15_repeated_opportunity":
            augmentation_sources=set(spec.get("augmentation_sources",[]))
            prior=dev[
                dev.market.eq(market)&dev.source.isin(augmentation_sources)&dev.y.notna()
            ].copy()
            prior_features=model_frame(prior) if not prior.empty else None
            prior_labels=prior["y"].astype(int) if not prior.empty else None
            pipeline.fit_with_prior(
                features,part["y"].astype(int),prior_features,prior_labels
            )
        else:
            pipeline.fit(features,part["y"].astype(int))
        pipelines[market]=pipeline
    model={
        "kind":"market_ensemble","by_market":pipelines,
        "decision_cutoff_by_market":dict(cfg.decision_cutoff_by_market),
        "invert_probability_by_market":dict(cfg.invert_probability_by_market),
    }
    joblib.dump(model,FROZEN_MODEL)

    obj={
        "app":APP,"version":VERSION,"created_at":now_iso(),
        "config":asdict(cfg),
        "seal_start":min(pd.Timestamp(v) for v in starts.values()).isoformat(),
        "seal_start_by_market":{k:pd.Timestamp(v).isoformat() for k,v in starts.items()},
        "seal_source_by_market":dict(cfg.fixed_seal_sources),
        "seal_end_by_market":{
            market:pd.Timestamp(raw_end).isoformat()
            for market,_,raw_end in cfg.fixed_seal_windows
        },
        "selected_model":win["spec_by_market"],
        "selected_threshold":win["threshold"],
        "dev_metrics":win["metrics"],
        "all_dev_results":results,
        "code_sha256":code_sha(),
        "prepare_code_sha256":sha256_file(ROOT/"prepare_real_data.py"),
        "event_data_sha256":sha256_file(EVENT_FILE),
        "source_status_sha256":sha256_file(SOURCE_STATUS_FILE),
        "data_sha256":sha256_file(LABELED_FILE),
        "model_sha256":sha256_file(FROZEN_MODEL),
    }
    obj["frozen_spec_sha256"]=sha256_json(obj)
    atomic_json(FROZEN_SPEC,obj)
    (OUT/"DEV_SEARCH_RESULTS.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    return obj

# Python conditional above would be invalid; use helper
def cert_checks(m:dict,b:dict,cfg:Config):
    us=m["by_market"].get("US",{"n":0,"balanced_accuracy":float("nan")})
    kr=m["by_market"].get("KR",{"n":0,"balanced_accuracy":float("nan")})
    raw=[
        ("total_events",m["n"]>=cfg.cert_min_total_events,m["n"],cfg.cert_min_total_events),
        ("us_events",us["n"]>=cfg.cert_min_us_events,us["n"],cfg.cert_min_us_events),
        ("kr_events",kr["n"]>=cfg.cert_min_kr_events,kr["n"],cfg.cert_min_kr_events),
        ("balanced_accuracy",m["balanced_accuracy"]>=cfg.cert_min_bal_acc,m["balanced_accuracy"],cfg.cert_min_bal_acc),
        ("auc",m["auc"]>=cfg.cert_min_auc,m["auc"],cfg.cert_min_auc),
        ("edge_vs_naive",m["edge_vs_naive"]>=cfg.cert_min_edge_vs_naive,m["edge_vs_naive"],cfg.cert_min_edge_vs_naive),
        ("highconf_coverage",m["highconf_coverage"]>=cfg.cert_min_highconf_coverage,m["highconf_coverage"],cfg.cert_min_highconf_coverage),
        ("highconf_accuracy",m["highconf_accuracy"]>=cfg.cert_min_highconf_acc,m["highconf_accuracy"],cfg.cert_min_highconf_acc),
        ("strategy_mean_signed_net",m["strategy_mean_signed_net"]>=cfg.cert_min_mean_signed_net,m["strategy_mean_signed_net"],cfg.cert_min_mean_signed_net),
        ("boot_bal_lower95",b["balanced_accuracy_lower95"]>=cfg.cert_min_boot_bal_lower95,b["balanced_accuracy_lower95"],cfg.cert_min_boot_bal_lower95),
        ("US_bal_acc",np.isfinite(us["balanced_accuracy"]) and us["balanced_accuracy"]>=cfg.cert_min_market_bal_acc,us["balanced_accuracy"],cfg.cert_min_market_bal_acc),
        ("KR_bal_acc",np.isfinite(kr["balanced_accuracy"]) and kr["balanced_accuracy"]>=cfg.cert_min_market_bal_acc,kr["balanced_accuracy"],cfg.cert_min_market_bal_acc),
    ]
    out=[]
    for n,ok,a,t in raw:
        if isinstance(a,(float,np.floating)) and not np.isfinite(a):
            av=None
        elif isinstance(a,(np.integer,int)):
            av=int(a)
        else:
            av=float(a)
        out.append({"name":n,"pass":bool(ok),"actual":av,"threshold":float(t) if isinstance(t,float) else int(t)})
    return out

def run_seal(df:pd.DataFrame,cfg:Config):
    if SEALED_JSON.exists() or SEALED_TXT.exists():
        raise RuntimeError("V3 SEAL은 이미 열렸습니다. 덮어쓰기 금지.")
    if not FROZEN_SPEC.exists() or not FROZEN_MODEL.exists():
        raise RuntimeError("먼저 --search 실행.")
    fr=json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    if code_sha()!=fr["code_sha256"]:
        raise RuntimeError("CODE SHA mismatch")
    if sha256_file(ROOT/"prepare_real_data.py")!=fr["prepare_code_sha256"]:
        raise RuntimeError("PREPARE CODE SHA mismatch")
    if sha256_file(EVENT_FILE)!=fr["event_data_sha256"]:
        raise RuntimeError("EVENT DATA SHA mismatch")
    if sha256_file(SOURCE_STATUS_FILE)!=fr["source_status_sha256"]:
        raise RuntimeError("SOURCE STATUS SHA mismatch")
    if sha256_file(LABELED_FILE)!=fr["data_sha256"]:
        raise RuntimeError("DATA SHA mismatch")
    if sha256_file(FROZEN_MODEL)!=fr["model_sha256"]:
        raise RuntimeError("MODEL SHA mismatch")

    x=df.copy()
    x["event_time_utc"]=pd.to_datetime(x["event_time_utc"],utc=True)
    seal_sources=fr.get("seal_source_by_market")
    starts=fr.get("seal_start_by_market")
    if seal_sources:
        pieces=[]
        for market,source_name in seal_sources.items():
            part=x[x.market.eq(market)&x.source.eq(source_name)].copy()
            if part.empty:
                raise RuntimeError(f"sealed source empty: {market}/{source_name}")
            pieces.append(part)
        seal=pd.concat(pieces).sort_values("event_time_utc").copy()
    elif starts:
        pieces=[]
        for market,raw_start in starts.items():
            ss=pd.Timestamp(raw_start)
            if ss.tzinfo is None: ss=ss.tz_localize("UTC")
            else: ss=ss.tz_convert("UTC")
            source=x if market=="ALL" else x[x.market.eq(market)]
            part=source[source["event_time_utc"]>=ss]
            raw_end=fr.get("seal_end_by_market",{}).get(market)
            if raw_end:
                end=pd.Timestamp(raw_end)
                if end.tzinfo is None: end=end.tz_localize("UTC")
                else: end=end.tz_convert("UTC")
                part=part[part["event_time_utc"]<end]
            pieces.append(part)
        seal=pd.concat(pieces).sort_values("event_time_utc").copy()
    else:
        ss=pd.Timestamp(fr["seal_start"])
        if ss.tzinfo is None: ss=ss.tz_localize("UTC")
        else: ss=ss.tz_convert("UTC")
        seal=x[x["event_time_utc"]>=ss].copy()
    model=joblib.load(FROZEN_MODEL)
    eligible=predict_eligible_frozen(model,seal)
    seal=seal.loc[eligible].copy()
    if seal.empty:
        raise RuntimeError("frozen eligibility policy selected no sealed events")
    p=predict_proba_frozen(model,seal)
    pred=seal[["event_id","event_time_utc","market","ticker","headline","event_type","y","fwd_ret_30m"]].copy()
    pred["prob_up"]=p
    pred["pred"]=np.where(p>=0.5,"UP","DOWN")
    thr=float(fr["selected_threshold"])
    pred["confidence"]=np.maximum(p,1-p)
    pred["high_conf"]=pred["confidence"]>=thr
    met=evaluate(pred.rename(columns={"prob_up":"prob"}),thr,cfg.round_trip_cost)
    boot=bootstrap_by_day(pred.rename(columns={"prob_up":"prob"}),thr,cfg.round_trip_cost)
    checks=cert_checks(met,boot,cfg)
    status="PASS" if all(c["pass"] for c in checks) else "FAIL"

    obj={
        "app":APP,"version":VERSION,"opened_at":now_iso(),
        "status":status,
        "target":f"exact-timestamp bio news: +{cfg.entry_lag_min}m entry -> +{cfg.horizon_min}m direction",
        "seal_start":fr["seal_start"],
        "seal_start_by_market":fr.get("seal_start_by_market",{}),
        "seal_source_by_market":fr.get("seal_source_by_market",{}),
        "seal_end":seal["event_time_utc"].max().isoformat(),
        "seal_end_by_market":{
            market:g["event_time_utc"].max().isoformat()
            for market,g in seal.groupby("market")
        },
        "metrics":met,"bootstrap":boot,"checks":checks,
        "selected_model":fr["selected_model"],
        "selected_threshold":thr,
        "eligibility_policy":"all structurally usable US and KR events",
        "code_sha256":fr["code_sha256"],
        "prepare_code_sha256":fr["prepare_code_sha256"],
        "event_data_sha256":fr["event_data_sha256"],
        "source_status_sha256":fr["source_status_sha256"],
        "data_sha256":fr["data_sha256"],
        "model_sha256":fr["model_sha256"],
        "frozen_spec_sha256":fr["frozen_spec_sha256"],
    }
    obj["certificate_sha256"]=sha256_json(obj)
    atomic_json(SEALED_JSON,obj)
    shutil.copy2(ROOT/"bio_news_30m_v3.py",CERT/"bio_news_30m_v3_SEALED_SOURCE.py")
    shutil.copy2(ROOT/"prepare_real_data.py",CERT/"prepare_real_data_SEALED_SOURCE.py")
    shutil.copy2(SOURCE_STATUS_FILE,CERT/SOURCE_STATUS_FILE.name)
    pred.to_csv(CERT/"SEALED_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
    lines=[
        "="*86,
        f"{APP} {VERSION} SEALED CERTIFICATION",
        "="*86,
        f"STATUS: {status}",
        f"TARGET: {obj['target']}",
        f"SEAL: {obj['seal_start']} -> {obj['seal_end']}",
        f"TOTAL n={met['n']} | US={met['by_market'].get('US',{}).get('n',0)} | KR={met['by_market'].get('KR',{}).get('n',0)}",
        "-"*86,
    ]
    for c in checks:
        lines.append(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']:<25} actual={c['actual']} threshold={c['threshold']}")
    lines += [
        "-"*86,
        f"accuracy={met['accuracy']:.4f}",
        f"balanced_accuracy={met['balanced_accuracy']:.4f}",
        f"AUC={met['auc']:.4f}",
        f"naive={met['naive_accuracy']:.4f}",
        f"edge_vs_naive={met['edge_vs_naive']:.4f}",
        f"highconf={met['highconf_accuracy']:.4f} coverage={met['highconf_coverage']:.4f}",
        f"all_signal_mean_signed_net={met['mean_signed_net']:.4%}",
        f"highconf_strategy_mean_signed_net={met['strategy_mean_signed_net']:.4%}",
        f"bootstrap balanced acc 95%=[{boot['balanced_accuracy_lower95']:.4f},{boot['balanced_accuracy_upper95']:.4f}]",
        f"certificate_sha256={obj['certificate_sha256']}",
        "="*86,
    ]
    SEALED_TXT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    return obj




def synthetic_labeled(n:int=2400)->pd.DataFrame:
    rng=np.random.default_rng(SEED)
    start=pd.Timestamp("2020-01-02",tz="UTC")
    rows=[]
    # deliberately noisy but learnable relationship
    pos_templates=[
        "FDA approval positive phase 3 primary endpoint met",
        "license agreement milestone payment successful clinical trial",
        "positive topline results statistically significant",
        "품목허가 승인 임상 3상 유효성 목표 달성",
    ]
    neg_templates=[
        "complete response letter clinical hold serious adverse event",
        "phase 2 did not meet primary endpoint trial discontinued",
        "registered direct public offering dilution warrants",
        "임상 실패 목표 미달 유상증자 전환사채",
    ]
    neutral=[
        "quarterly financial results conference update",
        "corporate presentation and business update",
        "정기 공시 기업설명회 사업 업데이트",
    ]
    for i in range(n):
        market="US" if i%3 else "KR"
        ticker=(f"T{i%120:03d}" if market=="US" else f"{i%80:06d}")
        latent=rng.normal()
        kind=rng.choice(["pos","neg","neu"],p=[.36,.34,.30])
        if kind=="pos":
            text=rng.choice(pos_templates); effect=0.85
        elif kind=="neg":
            text=rng.choice(neg_templates); effect=-0.85
        else:
            text=rng.choice(neutral); effect=0
        pre=rng.normal(0,.012)
        logit=effect + 0.8*np.tanh(pre*30) + rng.normal(0,1.0)
        y=int(logit>0)
        # future return respects label noisily
        fret=(1 if y else -1)*abs(rng.normal(.006,.006))
        ts=start+pd.Timedelta(hours=6*i)
        body=text+" "+("fda clinical trial" if i%5==0 else "")
        full=text+" "+body
        rows.append({
            "event_id":f"SYN:{i}",
            "event_time_utc":ts,
            "market":market,"ticker":ticker,
            "source":"SEC" if market=="US" else "KIND",
            "event_type":infer_event_type(full),
            "headline":text,"body":body,"text":full,
            "fwd_ret_30m":fret,"y":y,
            "pre_ret_2":pre/8,"pre_ret_5":pre/4,"pre_ret_15":pre/2,"pre_ret_30":pre,"pre_ret_60":pre*1.2,
            "pre_vol_10":abs(rng.normal(.006,.002)),"pre_vol_30":abs(rng.normal(.008,.002)),
            "volume_ratio_5_30":abs(rng.normal(1.2,.35)),
            "range_5m":abs(rng.normal(.01,.004)),
            "minutes_from_open":rng.uniform(20,250),"minutes_to_close":rng.uniform(40,250),
            "event_positive_kw":count_kw(full,POSITIVE_KW),
            "event_negative_kw":count_kw(full,NEGATIVE_KW),
            "event_financing_kw":count_kw(full,FINANCING_KW),
            "event_trial_kw":count_kw(full,TRIAL_KW),
            "event_regulatory_kw":count_kw(full,REGULATORY_KW),
            "event_ma_kw":count_kw(full,MA_KW),
            "event_earnings_kw":count_kw(full,EARNINGS_KW),
            "headline_len":len(text),"body_len":len(body),
        })
    return pd.DataFrame(rows)

def self_test()->int:
    tests=[]
    seed=load_seed_universe()
    tests.append(("seed universe >=150",len(seed)>=150))
    tests.append(("both markets in universe",set(seed["market"])=={"US","KR"}))

    # timestamp/session tests
    us_event=pd.Timestamp("2026-06-15 15:00:00",tz="UTC") # 11am ET summer
    local,op,cl=session_bounds(us_event,"US")
    tests.append(("US session conversion",op<=local<=cl))
    kr_event=pd.Timestamp("2026-06-15 03:00:00",tz="UTC") # noon KST
    local2,op2,cl2=session_bounds(kr_event,"KR")
    tests.append(("KR session conversion",op2<=local2<=cl2))

    # model pipeline
    cfg=Config(
        cert_min_total_events=100,
        cert_min_us_events=50,
        cert_min_kr_events=30,
        dev_min_days=60,
        seal_min_days=30,
        seal_fraction=.20,
        fixed_seal_sources=(),
        fixed_seal_windows=(),
        cert_min_bal_acc=.50,
        cert_min_auc=.50,
        cert_min_edge_vs_naive=-.20,
        cert_min_highconf_coverage=.02,
        cert_min_highconf_acc=.50,
        cert_min_boot_bal_lower95=.45,
        cert_min_market_bal_acc=.48,
    )
    x=synthetic_labeled(2400)
    dev,seal,starts=split_dev_seal(x,cfg)
    no_overlap=all(
        dev.loc[dev.market.eq(m),"event_time_utc"].max() <
        seal.loc[seal.market.eq(m),"event_time_utc"].min()
        for m in starts
    )
    tests.append(("dev/seal no overlap",no_overlap))
    spec={"kind":"logit","C":.7,"class_weight":"balanced"}
    model=make_model(spec)
    model.fit(model_frame(dev),dev["y"])
    p=model.predict_proba(model_frame(seal))[:,1]
    z=seal[["event_id","event_time_utc","market","ticker","y","fwd_ret_30m"]].copy()
    z["prob"]=p
    met=evaluate(z,.56,cfg.round_trip_cost)
    tests.append(("probabilities bounded",bool(((p>=0)&(p<=1)).all())))
    tests.append(("balanced accuracy finite",np.isfinite(met["balanced_accuracy"])))
    tests.append(("AUC finite",np.isfinite(met["auc"])))
    tests.append(("both market metrics",set(met["by_market"])=={"US","KR"}))
    boot=bootstrap_by_day(z,.56,cfg.round_trip_cost,nboot=80)
    tests.append(("bootstrap works",np.isfinite(boot["balanced_accuracy_lower95"])))

    passed=sum(bool(ok) for _,ok in tests)
    print("="*80)
    print(f"{APP} {VERSION} SELF TEST")
    print("="*80)
    for name,ok in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("-"*80)
    print(f"RESULT: {passed}/{len(tests)} PASS")
    print("SELF_TEST_STATUS="+("PASS" if passed==len(tests) else "FAIL"))
    return 0 if passed==len(tests) else 3




def latest_predict(features_csv:Path):
    if not FROZEN_MODEL.exists() or not FROZEN_SPEC.exists():
        raise RuntimeError("frozen model missing")
    x=pd.read_csv(features_csv)
    model=joblib.load(FROZEN_MODEL)
    x["eligible"]=predict_eligible_frozen(model,x)
    p=predict_proba_frozen(model,x)
    x["prob_up_30m"]=p
    x["direction"]=np.where(p>=.5,"UP","DOWN")
    x["confidence"]=np.maximum(p,1-p)
    fr=json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    x["high_conf"]=x["eligible"]&(x["confidence"]>=float(fr["selected_threshold"]))
    x.to_csv(OUT/"LATEST_NEWS_SIGNALS.csv",index=False,encoding="utf-8-sig")
    print(x.sort_values("confidence",ascending=False).head(50).to_string(index=False))




def load_cfg()->Config:
    return Config()

def load_search_frame(cfg:Config)->pd.DataFrame:
    """Load DEV outcomes while keeping fixed-source SEAL outcomes unparsed."""
    metadata=pd.read_csv(
        LABELED_FILE,compression="gzip",
        usecols=["event_id","market","source","event_time_utc"],
    )
    metadata["event_time_utc"]=pd.to_datetime(metadata.event_time_utc,utc=True)
    seal_pairs=set(cfg.fixed_seal_sources)
    is_seal=np.asarray([
        (str(market),str(source)) in seal_pairs
        for market,source in zip(metadata.market,metadata.source)
    ],dtype=bool)
    dev_rows=set(np.flatnonzero(~is_seal).tolist())
    development=pd.read_csv(
        LABELED_FILE,compression="gzip",parse_dates=["event_time_utc"],
        dtype={"ticker":str},low_memory=False,
        skiprows=lambda line_number: line_number>0 and (line_number-1) not in dev_rows,
    )
    # Search needs sealed source counts/time bounds only.  Outcomes and returns
    # remain absent until the explicit one-shot --seal command.
    sealed_metadata=metadata.loc[is_seal].copy()
    return pd.concat([development,sealed_metadata],ignore_index=True,sort=False)

def main()->int:
    ap=argparse.ArgumentParser(description=APP)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--build-universe",action="store_true")
    ap.add_argument("--collect-sec",action="store_true")
    ap.add_argument("--kind-csv",type=str,default="")
    ap.add_argument("--news-csv",type=str,default="")
    ap.add_argument("--combine-events",action="store_true")
    ap.add_argument("--label",action="store_true")
    ap.add_argument("--search",action="store_true")
    ap.add_argument("--seal",action="store_true")
    ap.add_argument("--latest-features",type=str,default="")
    args=ap.parse_args()
    cfg=load_cfg()

    if args.self_test:
        return self_test()

    universe=build_live_universe(cfg.min_universe) if args.build_universe else (
        pd.read_csv(DATA/"universe_live.csv",dtype={"ticker":str})
        if (DATA/"universe_live.csv").exists() else load_seed_universe()
    )

    frames=[]
    if args.collect_sec:
        sec=SECCollector(cfg).collect(universe)
        sec.to_csv(DATA/"events_sec.csv",index=False,encoding="utf-8-sig")
        print(f"[SEC] {len(sec)} events")
        frames.append(sec)
    if args.kind_csv:
        k=normalize_kind_csv(Path(args.kind_csv))
        k.to_csv(DATA/"events_kind.csv",index=False,encoding="utf-8-sig")
        print(f"[KIND] {len(k)} events")
        frames.append(k)
    if args.news_csv:
        n=normalize_generic_news_csv(Path(args.news_csv))
        n.to_csv(DATA/"events_generic.csv",index=False,encoding="utf-8-sig")
        print(f"[GENERIC] {len(n)} events")
        frames.append(n)

    if args.combine_events:
        for p in (DATA/"events_sec.csv",DATA/"events_kind.csv",DATA/"events_generic.csv"):
            if p.exists():
                frames.append(pd.read_csv(p))
        ev=combine_events(frames,cfg)
        print(f"[EVENTS] combined exact material events={len(ev)}")

    if args.label:
        ev=pd.read_csv(EVENT_FILE,compression="gzip")
        ev["event_time_utc"]=pd.to_datetime(ev["event_time_utc"],utc=True)
        x=build_labeled(ev,cfg)
        print(f"[LABELED] n={len(x)} US={(x.market=='US').sum()} KR={(x.market=='KR').sum()}")

    if args.search:
        x=load_search_frame(cfg)
        search_and_freeze(x,cfg)

    if args.seal:
        x=pd.read_csv(LABELED_FILE,compression="gzip",parse_dates=["event_time_utc"])
        run_seal(x,cfg)

    if args.latest_features:
        latest_predict(Path(args.latest_features))

    if not any([
        args.build_universe,args.collect_sec,args.kind_csv,args.news_csv,args.combine_events,
        args.label,args.search,args.seal,args.latest_features
    ]):
        ap.print_help()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
