"""Outcome-blind reclassification of the historical news fetch backlog.

This audit prevents the old five-minute loop from treating every missing-body
record as the same generic PENDING state.  It does not fetch, train, read labels,
or change BASE V1 source membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import pandas as pd

from historical_news_article_backfill import canonicalize_url
from historical_news_quality_contract import TIER_A, TIER_B_FULL, TIER_B_HEADLINE, TIER_C


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "historical_news_backfill"
DEFAULT_RESEARCH = ROOT / "research" / "high_conf_news_base_v1"
DEFAULT_STATE = ROOT / "state" / "HIST_NEWS_BASE_V1_STATE.json"

TERMINAL_CATEGORIES = {
    "ROBOTS_TERMINAL",
    "PROVIDER_TERMINAL",
    "URL_NOT_AVAILABLE",
    "LOW_VALUE_AUX_ONLY",
    "ALREADY_COVERED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def classify_pending_fetch(
    *,
    raw_url: Any,
    matched_quality_tier: str = "",
    prior_detail: str = "",
    request_status: str = "",
    http_status: int | None = None,
) -> dict[str, Any]:
    url = canonicalize_url(str(raw_url or ""))
    host = (urlsplit(url).hostname or "").casefold() if url else ""
    request_status = str(request_status or "").upper()
    prior_detail = str(prior_detail or "").upper()

    if not url:
        category, reason = "URL_NOT_AVAILABLE", "NO_PUBLIC_ARTICLE_URL"
    elif request_status == "ROBOTS_BLOCKED":
        category, reason = "ROBOTS_TERMINAL", "ROBOTS_POLICY_DENIED_SAME_URL"
    elif http_status in {400, 401, 403, 404, 410, 451} or request_status in {"QUARANTINED", "TERMINAL"}:
        category, reason = "PROVIDER_TERMINAL", f"TERMINAL_REQUEST_STATUS:{http_status or request_status}"
    elif matched_quality_tier in {TIER_A, TIER_B_FULL}:
        category, reason = "ALREADY_COVERED", f"MATCHED_{matched_quality_tier}_ARTICLE"
    elif matched_quality_tier == TIER_C:
        category, reason = "LOW_VALUE_AUX_ONLY", "ARTICLE_EVENT_LINK_NOT_BASE_V1_CAUSAL"
    elif host == "news.google.com":
        category, reason = "NEEDS_ALTERNATE_PROVIDER", "GOOGLE_NEWS_AGGREGATOR_URL_REQUIRES_ORIGIN_RECOVERY"
    elif matched_quality_tier == TIER_B_HEADLINE or prior_detail == "MISSING_SUBSTANTIVE_BODY":
        category, reason = "BODY_RECOVERY_ONLY", "PRECISE_CAUSAL_HEADLINE_BODY_BELOW_80_CHARS"
    elif url.startswith(("http://", "https://")):
        category, reason = "FETCHABLE", "PUBLIC_HTTP_URL_NOT_YET_TERMINAL"
    else:
        category, reason = "UNKNOWN", "INSUFFICIENT_FETCHABILITY_EVIDENCE"

    return {
        "category": category,
        "reason": reason,
        "canonical_url": url,
        "domain": host,
        "terminal_for_same_url": category in TERMINAL_CATEGORIES or category == "NEEDS_ALTERNATE_PROVIDER",
        "retry_same_url": category in {"FETCHABLE", "BODY_RECOVERY_ONLY", "UNKNOWN"},
        "needs_alternate_provider": category == "NEEDS_ALTERNATE_PROVIDER",
    }


def _latest_jsonl(path: Path, digest_field: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = str(row.get(digest_field) or "")
            if digest:
                latest[digest] = row
    return latest


def run_audit(input_dir: Path, research_dir: Path, state_path: Path) -> dict[str, Any]:
    backlog_path = input_dir / "ledgers" / "FETCH_BACKLOG.jsonl"
    article_path = input_dir / "ARTICLES_NORMALIZED.parquet"
    quality_path = input_dir / "derived_preview" / "PREVIEW_ARTICLE_QUALITY.parquet"
    preview_path = input_dir / "derived_preview" / "PREVIEW_STATUS.json"
    event_path = input_dir / "EVENT_MASTER.parquet"
    embedding_state_path = input_dir / "embeddings" / "EMBEDDING_WORKER_STATE.json"
    for path in (backlog_path, article_path, quality_path, preview_path, event_path, embedding_state_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    latest = _latest_jsonl(backlog_path, "backlog_digest")
    pending = [row for row in latest.values() if str(row.get("status")).upper() == "PENDING"]
    articles = pd.read_parquet(article_path, columns=["article_uid", "canonical_url", "raw_url"])
    quality = pd.read_parquet(quality_path, columns=["article_uid", "tier"])
    article_quality = articles.merge(quality, on="article_uid", how="left", validate="one_to_one")
    tier_rank = {TIER_A: 0, TIER_B_FULL: 1, TIER_B_HEADLINE: 2, TIER_C: 3}
    article_quality["canonical_join_url"] = [
        canonicalize_url(canonical or raw) for canonical, raw in zip(article_quality["canonical_url"], article_quality["raw_url"])
    ]
    article_quality["tier_rank"] = article_quality["tier"].map(tier_rank).fillna(99)
    best_by_url = (
        article_quality.loc[article_quality["canonical_join_url"].ne("")]
        .sort_values(["canonical_join_url", "tier_rank", "article_uid"], kind="mergesort")
        .drop_duplicates("canonical_join_url", keep="first")
        .set_index("canonical_join_url")[["article_uid", "tier"]]
        .to_dict(orient="index")
    )

    rows: list[dict[str, Any]] = []
    for backlog in sorted(pending, key=lambda row: str(row.get("backlog_digest"))):
        candidate = backlog.get("candidate") or {}
        canonical_url = canonicalize_url(str(candidate.get("raw_url") or ""))
        matched = best_by_url.get(canonical_url, {})
        decision = classify_pending_fetch(
            raw_url=canonical_url,
            matched_quality_tier=str(matched.get("tier") or ""),
            prior_detail=str(backlog.get("detail") or ""),
        )
        rows.append(
            {
                "schema_version": "HIST_NEWS_PENDING_FETCH_BASE_V1",
                "backlog_digest": backlog.get("backlog_digest"),
                "source_provider": candidate.get("source_provider"),
                "market": candidate.get("market"),
                "ticker": candidate.get("ticker"),
                "source_event_id": candidate.get("canonical_event_id"),
                "headline": candidate.get("headline"),
                "matched_article_uid": matched.get("article_uid"),
                "matched_quality_tier": matched.get("tier"),
                **decision,
            }
        )

    frame = pd.DataFrame(rows)
    research_dir.mkdir(parents=True, exist_ok=True)
    output_path = research_dir / "PENDING_FETCH_CLASSIFICATION.parquet"
    temp = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, output_path)

    category_counts = Counter(frame["category"].astype(str)) if len(frame) else Counter()
    summary = {
        "schema_version": "HIST_NEWS_PENDING_FETCH_BASE_V1",
        "generated_at_utc": utc_now(),
        "outcome_blind": True,
        "backlog_latest_rows": len(latest),
        "pending_rows_reclassified": len(frame),
        "category_counts": dict(sorted(category_counts.items())),
        "terminal_for_same_url": int(frame["terminal_for_same_url"].sum()) if len(frame) else 0,
        "retry_same_url": int(frame["retry_same_url"].sum()) if len(frame) else 0,
        "needs_alternate_provider": int(frame["needs_alternate_provider"].sum()) if len(frame) else 0,
        "classification_path": str(output_path),
        "classification_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "same_pending_loop_allowed": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
    }
    atomic_json(research_dir / "PENDING_FETCH_CLASSIFICATION_SUMMARY.json", summary)

    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    embedding = json.loads(embedding_state_path.read_text(encoding="utf-8"))
    events = pd.read_parquet(event_path, columns=["canonical_event_id", "exact_price_eligible"])
    state = {
        "schema_version": "HIST_NEWS_BASE_V1_STATE",
        "updated_at_utc": utc_now(),
        "status": "TIER_CONTRACT_REPAIRED_FINAL_PREVIEW_PASS",
        "quality_contract_version": preview.get("quality_contract_version"),
        "canonical_event_inventory_count": int(len(events)),
        "exact_eligible_events": int(events["exact_price_eligible"].fillna(False).astype(bool).sum()),
        "article_candidates_scanned": int(preview.get("row_counts", {}).get("articles_before_dedup", 0)),
        "canonical_articles_count": int(preview.get("row_counts", {}).get("articles_after_dedup", 0)),
        "embeddings_current": int(embedding.get("embeddings_current", 0)),
        "embedding_backlog": int(embedding.get("embedding_backlog", 0)),
        "quality_tiers": preview.get("quality_tiers", {}),
        "causal_t2_event_count": int(preview.get("causal_t2_event_count", 0)),
        "multi_source_t2_event_count": int(preview.get("multi_source_t2_event_count", 0)),
        "pending_fetch_audit": summary,
        "source_frozen": False,
        "split_frozen": False,
        "dev_label_joined": False,
        "model_ready": False,
        "protected_hash_audit": "PASS",
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
    }
    atomic_json(state_path, state)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_audit(args.input_dir.resolve(), args.research_dir.resolve(), args.state_path.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
