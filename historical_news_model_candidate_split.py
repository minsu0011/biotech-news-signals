"""Freeze a label-blind chronological split for exact causal NEWS candidates.

The global BASE V1 split covers the full 235k inventory.  Because that inventory
is dominated by recent non-priced discovery rows, its folds need not distribute
the exact-price, causal A/B_FULL model candidates.  This second split is frozen
after source membership but before the DEV label join and uses no outcome or
direction-prediction file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1"
RESEARCH = ROOT / "research" / "high_conf_news_base_v1"
OUTPUT = RESEARCH / "HIST_NEWS_MODEL_CANDIDATE_SPLIT_MANIFEST.parquet"
FREEZE = RESEARCH / "MODEL_CANDIDATE_SPLIT_FREEZE.json"
FOLD_COUNT = 5
NEWS_SOURCES = ("US_NEWS", "KR_NEWS")


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


class CandidateSplitError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateSplitError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    work = frame.copy()
    for row in work.to_dict(orient="records"):
        payload = {}
        for key, value in sorted(row.items()):
            if isinstance(value, pd.Timestamp):
                payload[key] = value.isoformat()
            elif pd.isna(value):
                payload[key] = None
            else:
                payload[key] = value
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def build_candidate_split() -> tuple[pd.DataFrame, dict[str, Any]]:
    freeze_path = DATA / "SOURCE_FREEZE.json"
    event_path = DATA / "EVENT_MASTER.parquet"
    causal_path = DATA / "CAUSAL_EVENT_NEWS.parquet"
    article_path = DATA / "ARTICLES_NORMALIZED.parquet"
    require(freeze_path.is_file() and event_path.is_file() and causal_path.is_file() and article_path.is_file(), "BASE V1 source freeze is incomplete")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    require(bool(freeze.get("source_membership_frozen")), "source membership is not frozen")
    require(bool(freeze.get("outcome_blind")), "source freeze is not outcome-blind")
    require(not (DATA / "POST_FREEZE_LABEL_JOIN_STATUS.json").exists(), "candidate split must freeze before DEV label join")
    require(not (DATA / "HIST_HIGH_CONF_NEWS_BASE_V1_LABELED.parquet").exists(), "labeled table already exists")

    events = pd.read_parquet(
        event_path,
        columns=[
            "canonical_event_id", "event_group_id", "source_event_group_id", "market", "ticker", "source_family",
            "event_time_utc", "exact_price_eligible",
        ],
    )
    causal = pd.read_parquet(
        causal_path,
        columns=["canonical_event_id", "tierA_article_count_t2", "tierBFull_article_count_t2"],
    )
    frame = events.merge(causal, on="canonical_event_id", how="left", validate="one_to_one")
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True, errors="raise")
    mask = (
        frame["source_family"].isin(NEWS_SOURCES)
        & frame["exact_price_eligible"].fillna(False).astype(bool)
        & (frame[["tierA_article_count_t2", "tierBFull_article_count_t2"]].fillna(0).sum(axis=1) > 0)
    )
    candidates = frame.loc[mask].copy()
    require(len(candidates) >= 100, "exact causal model-candidate support unexpectedly small")
    candidate_ids = set(candidates["canonical_event_id"].astype(str))
    union = UnionFind(sorted(candidate_ids))
    for _, group in candidates.loc[candidates["source_event_group_id"].astype(str).ne("")].groupby("source_event_group_id", sort=False):
        members = group["canonical_event_id"].astype(str).tolist()
        for other in members[1:]:
            union.union(members[0], other)
    article_links = pd.read_parquet(
        article_path, columns=["canonical_event_id", "syndication_root_id"]
    )
    article_links = article_links.loc[article_links["canonical_event_id"].astype(str).isin(candidate_ids)]
    article_links = article_links.loc[article_links["syndication_root_id"].astype(str).ne("")]
    for _, group in article_links.groupby("syndication_root_id", sort=False):
        members = sorted(set(group["canonical_event_id"].astype(str)))
        for other in members[1:]:
            union.union(members[0], other)
    components: dict[str, list[str]] = {}
    for event_id in sorted(candidate_ids):
        components.setdefault(union.find(event_id), []).append(event_id)
    model_group_by_event = {}
    for members in components.values():
        model_group = "meg_" + hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:20]
        for event_id in members:
            model_group_by_event[event_id] = model_group
    candidates["model_event_group_id"] = candidates["canonical_event_id"].map(model_group_by_event)
    fold_by_event: dict[str, int] = {}
    block_by_event: dict[str, str] = {}
    for source, source_rows in candidates.groupby("source_family", sort=True):
        source_rows = source_rows.sort_values(["event_time_utc", "canonical_event_id"], kind="mergesort").reset_index(drop=True)
        ids = source_rows["canonical_event_id"].astype(str).tolist()
        index = {event_id: position for position, event_id in enumerate(ids)}
        intervals = []
        for _, group in source_rows.groupby("model_event_group_id", sort=False):
            positions = [index[str(event_id)] for event_id in group["canonical_event_id"]]
            intervals.append((min(positions), max(positions)))
        intervals.sort()
        blocks: list[tuple[int, int]] = []
        for start, end in intervals:
            if blocks and start <= blocks[-1][1]:
                old_start, old_end = blocks[-1]
                blocks[-1] = (old_start, max(old_end, end))
            else:
                blocks.append((start, end))
        total = max(len(source_rows), 1)
        seen = 0
        for block_number, (start, end) in enumerate(blocks):
            size = end - start + 1
            midpoint = seen + size / 2
            fold = min(FOLD_COUNT - 1, int(midpoint * FOLD_COUNT / total))
            block_id = hashlib.sha256(
                f"{source}|{block_number}|{'|'.join(ids[start:end + 1])}".encode("utf-8")
            ).hexdigest()[:24]
            for event_id in ids[start : end + 1]:
                fold_by_event[event_id] = fold
                block_by_event[event_id] = block_id
            seen += size
    candidates["model_source_fold"] = candidates["canonical_event_id"].map(fold_by_event).astype(int)
    candidates["model_chronological_block_id"] = candidates["canonical_event_id"].map(block_by_event)
    candidates["candidate_contract"] = "EXACT_PRICE_ELIGIBLE_AND_CAUSAL_TIER_A_OR_B_FULL"
    manifest = candidates[
        [
            "canonical_event_id", "model_event_group_id", "source_event_group_id", "market", "ticker", "source_family", "event_time_utc",
            "model_source_fold", "model_chronological_block_id", "candidate_contract",
        ]
    ].sort_values(["source_family", "event_time_utc", "canonical_event_id"], kind="mergesort").reset_index(drop=True)

    require(manifest.groupby("model_event_group_id")["model_source_fold"].nunique().max() == 1, "event group crosses model folds")
    for source, source_rows in manifest.groupby("source_family"):
        require(source_rows["model_source_fold"].nunique() >= 4, f"too few model folds for {source}")
        ranges = source_rows.groupby("model_source_fold")["event_time_utc"].agg(["min", "max"]).sort_index()
        prior_max = None
        for row in ranges.itertuples():
            if prior_max is not None:
                require(row.min >= prior_max, f"model fold chronology overlaps for {source}")
            prior_max = row.max

    payload = {
        "version": "HIST_NEWS_MODEL_CANDIDATE_SPLIT_V1",
        "source_freeze_sha256": sha256_file(freeze_path),
        "event_master_sha256": sha256_file(event_path),
        "causal_event_news_sha256": sha256_file(causal_path),
        "articles_normalized_sha256": sha256_file(article_path),
        "manifest_semantic_sha256": semantic_sha(manifest),
        "candidate_rows": int(len(manifest)),
        "by_source": {str(k): int(v) for k, v in manifest["source_family"].value_counts().items()},
        "by_source_fold": {
            f"{source}|{int(fold)}": int(count)
            for (source, fold), count in manifest.groupby(["source_family", "model_source_fold"]).size().items()
        },
        "outcome_columns_read": [],
        "direction_predictions_read": False,
        "ticker_embargo_days": 7,
        "ticker_embargo_implementation": "purge outer/inner training rows within 7 days before validation per ticker; no transitive ticker union",
        "label_blind": True,
        "source_membership_changed": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_prospective_touched": False,
    }
    return manifest, payload


def main() -> int:
    manifest, payload = build_candidate_split()
    RESEARCH.mkdir(parents=True, exist_ok=True)
    if OUTPUT.is_file() or FREEZE.is_file():
        require(OUTPUT.is_file() and FREEZE.is_file(), "partial candidate split freeze exists")
        existing = json.loads(FREEZE.read_text(encoding="utf-8"))
        require(existing.get("manifest_file_sha256") == sha256_file(OUTPUT), "existing candidate split integrity failure")
        print(json.dumps({"status": "MODEL_CANDIDATE_SPLIT_REUSED", **existing}, sort_keys=True))
        return 0
    temp = OUTPUT.with_name(f".{OUTPUT.name}.tmp_{os.getpid()}")
    manifest.to_parquet(temp, index=False, compression="zstd")
    temp.replace(OUTPUT)
    payload["manifest_path"] = str(OUTPUT.resolve())
    payload["manifest_file_sha256"] = sha256_file(OUTPUT)
    write_json(FREEZE, payload)
    print(json.dumps({"status": "MODEL_CANDIDATE_SPLIT_FROZEN", **payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CandidateSplitError as exc:
        print(f"FAIL_CLOSED: {exc}")
        raise SystemExit(2)
