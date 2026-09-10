from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from historical_news_multisource_v2_body_recovery import salient_token_overlap, text_similarity


ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "research" / "multisource_v2" / "POLICY_FREEZE_MANIFEST.json"
QUEUE = ROOT / "research" / "hist_news_multisource_v2" / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
CURRENT = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING" / "V2_CURRENT_SOURCE_PROVENANCE.parquet"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
AUDIT = WORKING / "V2_LOCAL_KIND_CANDIDATES.parquet"
OFFICIAL = WORKING / "V2_RECOVERED_KIND_DISCLOSURES.parquet"
REPORT = ROOT / "research" / "multisource_v2" / "KIND_LINK_REPORT.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"
SOURCE_FILES = (
    ROOT / "data" / "events_kind_real.csv",
    ROOT / "data" / "events_kind_v15_expanded_raw.csv",
    ROOT / "data" / "events_kind_v17_raw.csv",
)
DISCOVERY_SECONDS = 24 * 60 * 60
DECISION_SECONDS = 2 * 60


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


def distinctive_codes(value: Any) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", str(value or ""))
    return {token.replace("-", "").casefold() for token in tokens if any(char.isdigit() for char in token)}


def kind_event_match(news_headline: Any, company: Any, disclosure_headline: Any, form: Any) -> dict[str, Any]:
    official_text = " ".join(map(str, (company or "", disclosure_headline or "", form or "")))
    overlap_count, overlap_ratio = salient_token_overlap(news_headline, official_text)
    sequence, char5_jaccard = text_similarity(news_headline, official_text)
    news_codes = distinctive_codes(news_headline)
    official_codes = distinctive_codes(official_text)
    code_consistent = not news_codes or bool(news_codes & official_codes)
    high_similarity = sequence >= 0.55 and char5_jaccard >= 0.10 and overlap_count >= 3
    high_coverage = overlap_count >= 4 and overlap_ratio >= 0.55 and char5_jaccard >= 0.03
    return {
        "headline_overlap_count": int(overlap_count),
        "headline_overlap_ratio": float(overlap_ratio),
        "headline_sequence_ratio": float(sequence),
        "headline_char5_jaccard": float(char5_jaccard),
        "news_distinctive_codes": ",".join(sorted(news_codes)),
        "official_distinctive_codes": ",".join(sorted(official_codes)),
        "distinctive_code_consistent": bool(code_consistent),
        "event_match_high": bool(code_consistent and (high_similarity or high_coverage)),
    }


def load_kind() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in SOURCE_FILES:
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype={"ticker": str}, low_memory=False)
        required = {"event_id", "ticker", "company", "event_time_utc", "form", "headline", "url"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"KIND source schema mismatch: {path.name}")
        frame = frame[list(required)].copy()
        frame["source_input_path"] = str(path.relative_to(ROOT))
        frame["source_input_sha256"] = sha256_file(path)
        parts.append(frame)
    source = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    if source.empty:
        return source
    source["ticker"] = source["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    source["event_time_utc"] = pd.to_datetime(source["event_time_utc"], utc=True, format="mixed", errors="coerce")
    source = source.dropna(subset=["event_id", "ticker", "event_time_utc"])
    source["_headline_length"] = source["headline"].fillna("").astype(str).str.len()
    source = source.sort_values(["event_id", "_headline_length", "source_input_path"], ascending=[True, False, True])
    return source.drop_duplicates("event_id", keep="first").reset_index(drop=True)


def build_candidates() -> pd.DataFrame:
    queue = pd.read_parquet(QUEUE)
    queue = queue.loc[queue["market"].eq("KR"), ["canonical_event_id", "market", "ticker", "event_time_utc"]].copy()
    queue["ticker"] = queue["ticker"].astype(str).str.zfill(6)
    queue["event_time_utc"] = pd.to_datetime(queue["event_time_utc"], utc=True, errors="raise")
    current = pd.read_parquet(CURRENT)[["canonical_event_id", "current_article_uid", "current_headline"]]
    queue = queue.merge(current, on="canonical_event_id", how="inner", validate="one_to_one")
    source = load_kind()
    if source.empty:
        return pd.DataFrame()
    candidates = queue.merge(source, on="ticker", suffixes=("_news", "_kind"), validate="many_to_many")
    candidates["seconds_from_news_event"] = (
        candidates["event_time_utc_kind"] - candidates["event_time_utc_news"]
    ).dt.total_seconds()
    candidates = candidates.loc[candidates["seconds_from_news_event"].abs().le(DISCOVERY_SECONDS)].copy()
    metrics = [
        kind_event_match(row.current_headline, row.company, row.headline, row.form)
        for row in candidates.itertuples(index=False)
    ]
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)


def run() -> dict[str, Any]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind") or policy.get("label_join_allowed"):
        raise RuntimeError("outcome-blind collection policy guard failed")
    candidates = build_candidates()
    if candidates.empty:
        audit = candidates
        accepted = candidates
    else:
        candidates["published_by_base_t2"] = candidates["seconds_from_news_event"].le(DECISION_SECONDS)
        candidates["accepted_official_root"] = candidates["event_match_high"] & candidates["published_by_base_t2"]
        candidates["status"] = "EVENT_MATCH_REJECTED"
        candidates.loc[candidates["event_match_high"], "status"] = "RETROSPECTIVE_AUX_POST_T2"
        candidates.loc[candidates["accepted_official_root"], "status"] = "CAUSAL_REGULATORY_ROOT_VERIFIED"
        candidates["outcome_fields_read"] = ""
        audit = candidates[[
            "canonical_event_id", "market", "ticker", "current_article_uid", "current_headline", "event_id",
            "event_time_utc_news", "event_time_utc_kind", "seconds_from_news_event", "company", "form", "headline",
            "url", "source_input_path", "source_input_sha256", "headline_overlap_count", "headline_overlap_ratio",
            "headline_sequence_ratio", "headline_char5_jaccard", "news_distinctive_codes",
            "official_distinctive_codes", "distinctive_code_consistent", "event_match_high", "published_by_base_t2",
            "accepted_official_root", "status", "outcome_fields_read",
        ]].sort_values(["canonical_event_id", "event_time_utc_kind", "event_id"], kind="mergesort").reset_index(drop=True)
        accepted = candidates.loc[candidates["event_match_high"]].copy()
    atomic_parquet(AUDIT, audit)

    rows: list[dict[str, Any]] = []
    for row in accepted.itertuples(index=False):
        body = " ".join(map(str, (row.company or "", row.form or "", row.headline or "")))
        rows.append({
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_RECOVERED_KIND_V1",
            "canonical_event_id": str(row.canonical_event_id), "market": "KR", "ticker": str(row.ticker),
            "issuer": str(row.company or ""), "event_time_utc": row.event_time_utc_news,
            "decision_cutoff_utc": row.event_time_utc_news + pd.Timedelta(minutes=2),
            "current_article_uid": str(row.current_article_uid), "current_origin_publisher": "",
            "candidate_source_uid": str(row.event_id), "source_type": "REGULATORY_PUBLIC_SOURCE",
            "url": str(row.url), "canonical_url": str(row.url), "hosting_domain": "kind.krx.co.kr",
            "publisher_canonical": "KRX KIND", "publisher_class": "REGULATORY",
            "publisher_resolution_method": "OFFICIAL_SOURCE_FILE",
            "published_at_utc": row.event_time_utc_kind,
            "timestamp_provenance": "KIND_ACCEPTANCE_TIMESTAMP_EXISTING_RAW_COLLECTION",
            "timestamp_precision": "EXACT_MINUTE", "timestamp_confidence": "HIGH",
            "headline": str(row.headline or ""), "body": body, "body_length": len(body),
            "headline_sequence_ratio": float(row.headline_sequence_ratio),
            "headline_char5_jaccard": float(row.headline_char5_jaccard),
            "body_sequence_ratio": 0.0, "body_char5_jaccard": 0.0,
            "salient_token_overlap_count": int(row.headline_overlap_count),
            "salient_token_overlap_ratio": float(row.headline_overlap_ratio),
            "event_match_high": True, "origin_distinct_from_current": True,
            "published_by_t2": bool(row.published_by_base_t2), "timestamp_acceptable_for_root": True,
            "likely_same_information_root": False, "source_root_type": "OFFICIAL_ROOT",
            "official_root_accepted": bool(row.accepted_official_root),
            "strict_independent_media_root_accepted": False,
            "raw_capture_path": str(row.source_input_path),
            "raw_capture_sha256": sha256_text(f"{row.event_id}\n{row.event_time_utc_kind}\n{body}"),
            "raw_mime_type": "text/csv-normalized-kind-record", "captured_at_utc": utc_now(),
            "outcome_fields_read": "", "status": str(row.status),
        })
    official = pd.DataFrame(rows)
    atomic_parquet(OFFICIAL, official)

    report = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_KIND_LINK_REPORT_V1", "generated_at_utc": utc_now(),
        "outcome_blind": True, "outcome_fields_read": [],
        "source_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in SOURCE_FILES if path.is_file()},
        "candidate_rows_within_24h": int(len(audit)),
        "candidate_events_within_24h": int(audit["canonical_event_id"].nunique()) if not audit.empty else 0,
        "semantic_match_rows": int(len(official)),
        "semantic_match_events": int(official["canonical_event_id"].nunique()) if not official.empty else 0,
        "accepted_causal_regulatory_events": int(official.loc[official["official_root_accepted"], "canonical_event_id"].nunique()) if not official.empty else 0,
        "official_media_within_abs_2m_events": int(audit.loc[audit["event_match_high"] & audit["seconds_from_news_event"].abs().le(DECISION_SECONDS), "canonical_event_id"].nunique()) if not audit.empty else 0,
        "network_requests": 0, "source_freeze_ready": False, "model_training_allowed": False,
        "base_v1_mutated": False, "research_seal_touched": False, "final_meta_touched": False, "v224_touched": False,
        "artifacts": {AUDIT.name: sha256_file(AUDIT), OFFICIAL.name: sha256_file(OFFICIAL)},
    }
    atomic_json(REPORT, report)
    if STATE.is_file():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state.update({
            "updated_at_utc": utc_now(), "phase": "OUTCOME_BLIND_KIND_OFFICIAL_SOURCE_LINKAGE",
            "kind_candidate_events": report["candidate_events_within_24h"],
            "kind_semantic_match_events": report["semantic_match_events"],
            "kind_causal_root_events": report["accepted_causal_regulatory_events"],
            "labels_read": False, "source_frozen": False, "model_training_allowed": False,
            "research_seal_touched": False, "final_meta_touched": False, "v224_touched": False,
        })
        atomic_json(STATE, state)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
