from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "research" / "multisource_v2" / "POLICY_FREEZE_MANIFEST.json"
QUEUE = ROOT / "research" / "hist_news_multisource_v2" / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
CURRENT = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING" / "V2_CURRENT_SOURCE_PROVENANCE.parquet"
BASE_ARTICLES = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1" / "ARTICLES_NORMALIZED.parquet"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
AUDIT = WORKING / "V2_LOCAL_OFFICIAL_CANDIDATES.parquet"
OFFICIAL = WORKING / "V2_RECOVERED_OFFICIAL_DISCLOSURES.parquet"
REPORT = ROOT / "research" / "multisource_v2" / "OFFICIAL_LINK_REPORT.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"

# These are collection-era raw/period files. DEV, seal, price-candidate, model and
# outcome-derived files are deliberately excluded from the source universe.
SOURCE_FILES = (
    ROOT / "data" / "events_sec_v9_2015_2017.csv",
    ROOT / "data" / "events_sec_v12_2018_raw.csv",
    ROOT / "data" / "events_sec_v7_early.csv",
    ROOT / "data" / "events_sec_v8_middle.csv",
)
ALLOWED_FORMS = {"8-K", "6-K", "10-Q", "10-K", "20-F", "40-F"}
DISCOVERY_WINDOW = pd.Timedelta(hours=24)
DECISION_LAG = pd.Timedelta(minutes=2)
WINDOW_TOKENS = 160
WINDOW_STRIDE = 40

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "will", "with", "update", "corrected", "rpt",
    "brief", "factbox", "exclusive", "sources", "says", "said", "company", "companies",
    "corp", "corporation", "inc", "incorporated", "plc", "ltd", "limited", "co", "new",
    "u", "s", "us",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def informative_tokens(value: Any, ticker: Any = "") -> list[str]:
    ticker_token = str(ticker or "").casefold()
    return [
        token for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(value or "").casefold())
        if len(token) >= 3 and token not in STOPWORDS and token != ticker_token and not token.isdigit()
    ]


def event_match_metrics(headline: Any, disclosure_body: Any, ticker: Any = "") -> dict[str, Any]:
    headline_tokens = set(informative_tokens(headline, ticker))
    body_tokens = informative_tokens(disclosure_body, ticker)
    best_count, best_coverage, best_jaccard = 0, 0.0, 0.0
    for start in range(0, max(1, len(body_tokens)), WINDOW_STRIDE):
        window = set(body_tokens[start:start + WINDOW_TOKENS])
        overlap = headline_tokens & window
        coverage = len(overlap) / len(headline_tokens) if headline_tokens else 0.0
        union = headline_tokens | window
        jaccard = len(overlap) / len(union) if union else 0.0
        candidate = (len(overlap), coverage, jaccard)
        if candidate > (best_count, best_coverage, best_jaccard):
            best_count, best_coverage, best_jaccard = candidate
    short_exact = len(headline_tokens) <= 6 and best_count >= 4 and best_coverage >= 0.80
    ordinary = best_count >= 5 and best_coverage >= 0.60
    accepted = bool(len(str(disclosure_body or "")) >= 80 and best_jaccard >= 0.04 and (short_exact or ordinary))
    return {
        "headline_informative_token_count": len(headline_tokens),
        "best_window_overlap_count": best_count,
        "best_window_headline_coverage": float(best_coverage),
        "best_window_token_jaccard": float(best_jaccard),
        "event_match_high": accepted,
    }


def load_official_source() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in SOURCE_FILES:
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        required = {"event_id", "ticker", "event_time_utc", "form", "headline", "body", "url"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"official source schema mismatch: {path.name}")
        frame = frame[list(required)].copy()
        frame["source_input_path"] = str(path.relative_to(ROOT))
        frame["source_input_sha256"] = sha256_file(path)
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    source = pd.concat(parts, ignore_index=True, sort=False)
    source["event_time_utc"] = pd.to_datetime(source["event_time_utc"], utc=True, errors="coerce")
    source = source.dropna(subset=["event_id", "ticker", "event_time_utc"])
    source = source.loc[source["form"].astype(str).isin(ALLOWED_FORMS)].copy()
    source["_body_length"] = source["body"].fillna("").astype(str).str.len()
    source = source.sort_values(["event_id", "_body_length", "source_input_path"], ascending=[True, False, True])
    return source.drop_duplicates("event_id", keep="first").reset_index(drop=True)


def build_candidates() -> pd.DataFrame:
    queue = pd.read_parquet(QUEUE)
    queue = queue.loc[queue["market"].eq("US"), ["canonical_event_id", "market", "ticker", "event_time_utc"]].copy()
    queue["event_time_utc"] = pd.to_datetime(queue["event_time_utc"], utc=True, errors="raise")
    current = pd.read_parquet(CURRENT)[["canonical_event_id", "current_article_uid"]]
    articles = pd.read_parquet(BASE_ARTICLES, columns=["article_uid", "headline"])
    queue = queue.merge(current, on="canonical_event_id", how="inner", validate="one_to_one")
    queue = queue.merge(articles, left_on="current_article_uid", right_on="article_uid", how="inner", validate="one_to_one")
    official = load_official_source()
    if official.empty:
        return pd.DataFrame()
    candidates = queue.merge(official, on="ticker", suffixes=("_news", "_official"), validate="many_to_many")
    candidates["seconds_from_news_event"] = (
        candidates["event_time_utc_official"] - candidates["event_time_utc_news"]
    ).dt.total_seconds()
    candidates = candidates.loc[candidates["seconds_from_news_event"].abs().le(DISCOVERY_WINDOW.total_seconds())].copy()
    metrics = [
        event_match_metrics(row.headline_news, row.body, row.ticker)
        for row in candidates.itertuples(index=False)
    ]
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)


def run() -> dict[str, Any]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind") or policy.get("label_join_allowed"):
        raise RuntimeError("outcome-blind collection policy guard failed")
    candidates = build_candidates()
    if candidates.empty:
        atomic_parquet(AUDIT, candidates)
        atomic_parquet(OFFICIAL, candidates)
    else:
        candidates["decision_cutoff_utc"] = candidates["event_time_utc_news"] + DECISION_LAG
        candidates["published_by_base_t2"] = candidates["event_time_utc_official"].le(candidates["decision_cutoff_utc"])
        candidates["accepted_official_root"] = (
            candidates["event_match_high"].fillna(False).astype(bool)
            & candidates["published_by_base_t2"].fillna(False).astype(bool)
        )
        candidates["status"] = "EVENT_MATCH_REJECTED"
        candidates.loc[candidates["event_match_high"], "status"] = "RETROSPECTIVE_AUX_POST_T2"
        candidates.loc[candidates["accepted_official_root"], "status"] = "CAUSAL_REGULATORY_ROOT_VERIFIED"
        candidates["outcome_fields_read"] = ""
        audit_columns = [
            "canonical_event_id", "market", "ticker", "current_article_uid", "headline_news", "event_id",
            "event_time_utc_news", "event_time_utc_official", "seconds_from_news_event", "form", "url",
            "source_input_path", "source_input_sha256", "headline_informative_token_count",
            "best_window_overlap_count", "best_window_headline_coverage", "best_window_token_jaccard",
            "event_match_high", "published_by_base_t2", "accepted_official_root", "status", "outcome_fields_read",
        ]
        audit = candidates[audit_columns].sort_values(
            ["canonical_event_id", "event_time_utc_official", "event_id"], kind="mergesort"
        ).reset_index(drop=True)
        atomic_parquet(AUDIT, audit)
        accepted = candidates.loc[candidates["event_match_high"]].copy()
        rows: list[dict[str, Any]] = []
        for row in accepted.itertuples(index=False):
            content_sha = sha256_text(f"{row.event_id}\n{row.event_time_utc_official}\n{row.body}")
            rows.append({
                "schema_version": "HIST_NEWS_MULTISOURCE_V2_RECOVERED_OFFICIAL_V1",
                "canonical_event_id": str(row.canonical_event_id), "market": "US", "ticker": str(row.ticker),
                "issuer": "", "event_time_utc": row.event_time_utc_news,
                "decision_cutoff_utc": row.decision_cutoff_utc,
                "current_article_uid": str(row.current_article_uid), "current_origin_publisher": "Reuters",
                "candidate_source_uid": str(row.event_id), "source_type": "REGULATORY_PUBLIC_SOURCE",
                "url": str(row.url), "canonical_url": str(row.url), "hosting_domain": "sec.gov",
                "publisher_canonical": "U.S. SEC", "publisher_class": "REGULATORY",
                "publisher_resolution_method": "OFFICIAL_SOURCE_FILE",
                "published_at_utc": row.event_time_utc_official,
                "timestamp_provenance": "SEC_ACCEPTANCE_TIMESTAMP_EXISTING_RAW_COLLECTION",
                "timestamp_precision": "EXACT_SECOND", "timestamp_confidence": "HIGH",
                "headline": str(row.headline_official), "body": str(row.body or ""),
                "body_length": len(str(row.body or "")), "headline_sequence_ratio": 0.0,
                "headline_char5_jaccard": float(row.best_window_token_jaccard),
                "body_sequence_ratio": 0.0, "body_char5_jaccard": 0.0,
                "salient_token_overlap_count": int(row.best_window_overlap_count),
                "salient_token_overlap_ratio": float(row.best_window_headline_coverage),
                "event_match_high": True, "origin_distinct_from_current": True,
                "published_by_t2": bool(row.published_by_base_t2), "timestamp_acceptable_for_root": True,
                "likely_same_information_root": False, "source_root_type": "OFFICIAL_ROOT",
                "official_root_accepted": bool(row.accepted_official_root),
                "strict_independent_media_root_accepted": False,
                "raw_capture_path": str(row.source_input_path), "raw_capture_sha256": content_sha,
                "raw_mime_type": "text/csv-normalized-sec-record",
                "captured_at_utc": utc_now(), "outcome_fields_read": "", "status": str(row.status),
            })
        official_frame = pd.DataFrame(rows)
        atomic_parquet(OFFICIAL, official_frame)

    audit = pd.read_parquet(AUDIT)
    official_frame = pd.read_parquet(OFFICIAL)
    accepted_count = int(official_frame["official_root_accepted"].fillna(False).astype(bool).sum()) if not official_frame.empty else 0
    report = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_OFFICIAL_LINK_REPORT_V1",
        "generated_at_utc": utc_now(), "outcome_blind": True, "outcome_fields_read": [],
        "source_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in SOURCE_FILES if path.is_file()},
        "candidate_rows_within_24h": int(len(audit)),
        "candidate_events_within_24h": int(audit["canonical_event_id"].nunique()) if not audit.empty else 0,
        "semantic_match_events": int(official_frame["canonical_event_id"].nunique()) if not official_frame.empty else 0,
        "accepted_causal_regulatory_events": accepted_count,
        "network_requests": 0, "source_freeze_ready": False, "model_training_allowed": False,
        "base_v1_mutated": False, "research_seal_touched": False, "final_meta_touched": False, "v224_touched": False,
        "artifacts": {AUDIT.name: sha256_file(AUDIT), OFFICIAL.name: sha256_file(OFFICIAL)},
    }
    atomic_json(REPORT, report)
    if STATE.is_file():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state.update({
            "updated_at_utc": utc_now(), "phase": "OUTCOME_BLIND_OFFICIAL_SOURCE_LINKAGE",
            "official_candidate_events": report["candidate_events_within_24h"],
            "official_semantic_match_events": report["semantic_match_events"],
            "official_causal_root_events": report["accepted_causal_regulatory_events"],
            "labels_read": False, "source_frozen": False, "model_training_allowed": False,
            "research_seal_touched": False, "final_meta_touched": False, "v224_touched": False,
        })
        atomic_json(STATE, state)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
