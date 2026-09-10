from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import pandas as pd

from historical_news_multisource_v2_body_recovery import char_ngrams, normalized_text, text_similarity


ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "research" / "multisource_v2" / "POLICY_FREEZE_MANIFEST.json"
EVENT_TIME_POLICY = ROOT / "research" / "multisource_v2" / "EVENT_TIME_RESOLUTION_POLICY_FREEZE.json"
QUEUE = ROOT / "research" / "hist_news_multisource_v2" / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
CURRENT = WORKING / "V2_CURRENT_SOURCE_PROVENANCE.parquet"
RECOVERED = WORKING / "V2_RECOVERED_PUBLIC_ARTICLES.parquet"
OFFICIAL_RECOVERED = WORKING / "V2_RECOVERED_OFFICIAL_DISCLOSURES.parquet"
KIND_RECOVERED = WORKING / "V2_RECOVERED_KIND_DISCLOSURES.parquet"
LOCAL = WORKING / "V2_LOCAL_CACHE_LINK_AUDIT.parquet"
RSS = WORKING / "V2_DISCOVERED_RSS_ITEMS.parquet"
BASE_ARTICLES = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1" / "ARTICLES_NORMALIZED.parquet"
EVENT_TIME_MANIFEST = WORKING / "V2_EVENT_TIME_MANIFEST.parquet"
SOURCE_GRAPH = WORKING / "V2_SOURCE_GRAPH.parquet"
SYNDICATION_ROOTS = WORKING / "V2_SYNDICATION_ROOTS.parquet"
CAUSAL_FEATURES = WORKING / "V2_CAUSAL_TPLUS2_FEATURES.parquet"
REPORT = ROOT / "research" / "multisource_v2" / "SOURCE_GRAPH_AUDIT.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"

MAJOR_CLASSES = {"MAJOR_WIRE", "MAJOR_FINANCIAL", "MAJOR_GENERAL"}
SPECIALIST_CLASSES = {"BIOTECH_SPECIALIST", "PHARMA_SPECIALIST"}
PRECISE_TIMESTAMPS = {"EXACT_SECOND", "EXACT_MINUTE"}
HIGH_TIMESTAMP_CONFIDENCE = {"HIGH", "HIGH_ORIGIN_PAGE"}
DETERMINISTIC_OFFICIAL_TIMESTAMP_CONFIDENCE = {"MEDIUM_DETERMINISTIC_IDENTIFIER"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *values: Any) -> str:
    material = json.dumps([str(value or "") for value in values], ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def host_of(url: Any) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def entropy(values: Iterable[Any]) -> float:
    counts = pd.Series(list(values), dtype="object").value_counts()
    if counts.empty:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-sum(float(p) * math.log(float(p)) for p in probabilities if p > 0))


def char5_containment(left: Any, right: Any) -> float:
    left_grams = char_ngrams(normalized_text(left))
    right_grams = char_ngrams(normalized_text(right))
    if not left_grams or not right_grams:
        return 0.0
    return float(len(left_grams & right_grams) / min(len(left_grams), len(right_grams)))


def shares_base_information_root(
    base_body: Any,
    candidate_body: Any,
    headline_sequence_ratio: Any,
    time_delta_seconds: Any,
) -> tuple[bool, str]:
    """Conservatively reject independence that cannot be proved against BASE V1.

    BASE V1 sometimes stores a short multi-item digest excerpt.  A long article from
    another publisher appearing within minutes with the same headline-level event
    cannot be demonstrated to be an independent information root from that excerpt.
    It is therefore joined to the frozen BASE root rather than counted twice.
    """
    sequence, jaccard = text_similarity(base_body, candidate_body)
    containment = char5_containment(base_body, candidate_body)
    if sequence >= 0.82 or jaccard >= 0.70 or containment >= 0.55:
        return True, "BASE_CONTENT_NEAR_DUPLICATE"
    base_length = len(normalized_text(base_body))
    candidate_length = len(normalized_text(candidate_body))
    try:
        headline_ratio = float(headline_sequence_ratio)
        delta_seconds = abs(float(time_delta_seconds))
    except (TypeError, ValueError):
        return False, ""
    if (
        0 < base_length < 300
        and candidate_length >= 500
        and headline_ratio >= 0.70
        and delta_seconds <= 300.0
    ):
        return True, "BASE_SHORT_EXCERPT_NEAR_SIMULTANEOUS_INDEPENDENCE_UNPROVEN"
    return False, ""


def build_event_time_manifest(events: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    confidence = recovered["timestamp_confidence"].astype(str) if not recovered.empty else pd.Series(dtype="object")
    root_type = recovered["source_root_type"].astype(str) if not recovered.empty else pd.Series(dtype="object")
    timestamp_allowed = (
        confidence.isin(HIGH_TIMESTAMP_CONFIDENCE)
        | (root_type.eq("OFFICIAL_ROOT") & confidence.isin(DETERMINISTIC_OFFICIAL_TIMESTAMP_CONFIDENCE))
    )
    eligible = recovered.loc[
        recovered["event_match_high"].fillna(False).astype(bool)
        & recovered["timestamp_precision"].astype(str).isin(PRECISE_TIMESTAMPS)
        & timestamp_allowed
        & recovered["source_root_type"].astype(str).isin({"OFFICIAL_ROOT", "MEDIA_REPORTING_ROOT"})
        & (
            recovered["official_root_accepted"].fillna(False).astype(bool)
            | recovered["strict_independent_media_root_accepted"].fillna(False).astype(bool)
        )
    ].copy() if not recovered.empty else recovered.copy()
    if not eligible.empty:
        eligible["published_at_utc"] = pd.to_datetime(eligible["published_at_utc"], utc=True, errors="coerce")
        eligible = eligible.dropna(subset=["published_at_utc"])

    rows: list[dict[str, Any]] = []
    grouped = {key: group for key, group in eligible.groupby("canonical_event_id", sort=False)} if not eligible.empty else {}
    for event in events.itertuples(index=False):
        base_time = pd.Timestamp(event.event_time_utc)
        candidates = grouped.get(str(event.canonical_event_id))
        resolved = base_time
        method = "PRESERVE_BASE_V1_EVENT_TIME"
        source_url = ""
        source_type = "BASE_V1_EVENT_TIME"
        source_confidence = "BASE_V1_FROZEN"
        if candidates is not None and not candidates.empty:
            earlier = candidates.loc[candidates["published_at_utc"] < base_time].sort_values(
                ["published_at_utc", "canonical_url"], kind="mergesort"
            )
            if not earlier.empty:
                chosen = earlier.iloc[0]
                resolved = pd.Timestamp(chosen["published_at_utc"])
                method = (
                    "EARLIEST_HIGH_CONFIDENCE_OFFICIAL_DISCLOSURE"
                    if str(chosen["source_root_type"]) == "OFFICIAL_ROOT"
                    else "EARLIEST_HIGH_CONFIDENCE_ORIGINAL_PUBLICATION"
                )
                source_url = str(chosen["canonical_url"])
                source_type = str(chosen["source_root_type"])
                source_confidence = str(chosen["timestamp_confidence"])
        rows.append({
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_EVENT_TIME_MANIFEST_V1",
            "canonical_event_id": str(event.canonical_event_id),
            "market": str(event.market),
            "ticker": str(event.ticker),
            "base_v1_event_time_utc": base_time,
            "v2_resolved_event_time_utc": resolved,
            "v2_decision_cutoff_utc": resolved + pd.Timedelta(minutes=2),
            "event_time_changed": bool(resolved != base_time),
            "resolution_method": method,
            "resolution_source_url": source_url,
            "resolution_source_type": source_type,
            "resolution_source_timestamp_confidence": source_confidence,
            "event_time_ambiguous": False,
            "outcome_fields_read": "",
        })
    return pd.DataFrame(rows)


def current_nodes(events: pd.DataFrame, event_times: pd.DataFrame) -> pd.DataFrame:
    provenance = pd.read_parquet(CURRENT)
    wanted = set(provenance["current_article_uid"].astype(str))
    base = pd.read_parquet(BASE_ARTICLES, columns=[
        "article_uid", "canonical_url", "syndication_root_id", "published_at_precision",
        "published_at_confidence", "source_type", "raw_capture_sha256",
    ])
    base = base.loc[base["article_uid"].astype(str).isin(wanted)].copy()
    frame = events.merge(provenance, on="canonical_event_id", how="inner", validate="one_to_one")
    frame = frame.merge(base, left_on="current_article_uid", right_on="article_uid", how="left", validate="one_to_one")
    frame = frame.merge(event_times[["canonical_event_id", "v2_resolved_event_time_utc", "v2_decision_cutoff_utc"]],
                        on="canonical_event_id", how="inner", validate="one_to_one")
    rows = []
    for row in frame.itertuples(index=False):
        published = pd.to_datetime(row.current_published_at_utc, utc=True, errors="coerce")
        precise = str(row.published_at_precision) in PRECISE_TIMESTAMPS
        causal = bool(pd.notna(published) and precise and published <= row.v2_decision_cutoff_utc)
        independent_root = str(row.syndication_root_id or "") or stable_id("MEDIA_ROOT_", row.current_article_uid)
        publisher = str(row.origin_publisher_resolved)
        rows.append({
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_SOURCE_GRAPH_V1",
            "source_node_id": stable_id("NODE_", row.canonical_event_id, row.current_article_uid),
            "canonical_event_id": str(row.canonical_event_id),
            "market": str(row.market),
            "ticker": str(row.ticker),
            "article_uid": str(row.current_article_uid),
            "source_kind": "BASE_V1_CURRENT_MEDIA",
            "origin_source_id": stable_id("ORIGIN_", publisher),
            "publisher_canonical": publisher,
            "publisher_class": str(row.publisher_class_resolved),
            "hosting_domain": host_of(row.canonical_url),
            "published_at_utc": published,
            "timestamp_precision": str(row.published_at_precision),
            "timestamp_confidence": str(row.published_at_confidence),
            "canonical_url": str(row.canonical_url or ""),
            "syndication_root_id": independent_root,
            "official_origin_id": "",
            "independent_reporting_root_id": independent_root,
            "source_root_type": "MEDIA_REPORTING_ROOT",
            "root_acceptance_status": "BASE_V1_MEDIA_ROOT_V2_CAUSAL" if causal else "BASE_V1_MEDIA_POST_V2_T2",
            "is_causal_v2_t2": causal,
            "counted_causal_information_root": causal,
            "counted_independent_media_root": causal,
            "counted_official_root": False,
            "is_major_media": str(row.publisher_class_resolved) in MAJOR_CLASSES,
            "is_specialist_media": str(row.publisher_class_resolved) in SPECIALIST_CLASSES,
            "event_match_evidence": "BASE_V1_FROZEN_EVENT_MEMBERSHIP",
            "provenance_evidence": str(row.raw_capture_sha256 or ""),
            "outcome_fields_read": "",
        })
    return pd.DataFrame(rows)


def independence_root_map(recovered: pd.DataFrame, event_times: pd.DataFrame) -> dict[int, tuple[str, bool]]:
    """Union near-duplicate recovered media articles within each event."""
    if recovered.empty:
        return {}
    cutoffs = event_times.set_index("canonical_event_id")["v2_decision_cutoff_utc"]
    frame = recovered.copy()
    frame["_published"] = pd.to_datetime(frame["published_at_utc"], utc=True, errors="coerce")
    frame["_cutoff"] = frame["canonical_event_id"].map(cutoffs)
    eligible = frame.loc[
        frame["strict_independent_media_root_accepted"].fillna(False).astype(bool)
        & frame["_published"].notna()
        & frame["timestamp_precision"].astype(str).isin(PRECISE_TIMESTAMPS)
        & frame["_published"].le(frame["_cutoff"])
    ]
    mapping: dict[int, tuple[str, bool]] = {}
    for event_id, group in eligible.groupby("canonical_event_id", sort=False):
        indices = list(group.index)
        parent = {index: index for index in indices}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        for position, left in enumerate(indices):
            for right in indices[position + 1:]:
                left_row, right_row = frame.loc[left], frame.loc[right]
                sequence, jaccard = text_similarity(left_row["body"], right_row["body"])
                containment = char5_containment(left_row["body"], right_row["body"])
                if (
                    str(left_row["raw_capture_sha256"]) == str(right_row["raw_capture_sha256"])
                    or sequence >= 0.82
                    or jaccard >= 0.70
                    or containment >= 0.55
                ):
                    union(left, right)
        components: dict[int, list[int]] = {}
        for index in indices:
            components.setdefault(find(index), []).append(index)
        for members in components.values():
            identities = sorted(str(frame.loc[index, "raw_capture_sha256"]) for index in members)
            root_id = stable_id("MEDIA_ROOT_", event_id, *identities)
            merged = len(members) > 1
            for index in members:
                mapping[int(index)] = (root_id, merged)
    return mapping


def base_recovered_root_map(
    recovered: pd.DataFrame,
    event_times: pd.DataFrame,
) -> dict[int, tuple[str, str]]:
    """Map recovered media to its BASE V1 information root when independence is unproved."""
    if recovered.empty:
        return {}
    provenance = pd.read_parquet(CURRENT, columns=[
        "canonical_event_id", "current_article_uid", "current_published_at_utc",
    ])
    wanted = set(provenance["current_article_uid"].astype(str))
    base = pd.read_parquet(BASE_ARTICLES, columns=[
        "article_uid", "body", "syndication_root_id", "raw_capture_sha256",
    ])
    base = base.loc[base["article_uid"].astype(str).isin(wanted)].copy()
    reference = provenance.merge(
        base, left_on="current_article_uid", right_on="article_uid", how="left", validate="one_to_one"
    ).set_index("canonical_event_id")
    cutoffs = event_times.set_index("canonical_event_id")["v2_decision_cutoff_utc"]
    mapping: dict[int, tuple[str, str]] = {}
    for index, row in recovered.iterrows():
        if not bool(row.get("strict_independent_media_root_accepted", False)):
            continue
        event_id = str(row.get("canonical_event_id", ""))
        if event_id not in reference.index or event_id not in cutoffs.index:
            continue
        published = pd.to_datetime(row.get("published_at_utc"), utc=True, errors="coerce")
        if pd.isna(published) or published > pd.Timestamp(cutoffs.loc[event_id]):
            continue
        base_row = reference.loc[event_id]
        base_published = pd.to_datetime(base_row.get("current_published_at_utc"), utc=True, errors="coerce")
        if pd.isna(base_published):
            continue
        base_sha = str(base_row.get("raw_capture_sha256") or "")
        candidate_sha = str(row.get("raw_capture_sha256") or "")
        if base_sha and candidate_sha and base_sha == candidate_sha:
            shared, reason = True, "BASE_RAW_CAPTURE_DUPLICATE"
        else:
            shared, reason = shares_base_information_root(
                base_row.get("body", ""),
                row.get("body", ""),
                row.get("headline_sequence_ratio", 0.0),
                (published - base_published).total_seconds(),
            )
        if not shared:
            continue
        root_id = str(base_row.get("syndication_root_id") or "") or stable_id(
            "MEDIA_ROOT_", base_row.get("current_article_uid", "")
        )
        mapping[int(index)] = (root_id, reason)
    return mapping


def recovered_nodes(recovered: pd.DataFrame, event_times: pd.DataFrame) -> pd.DataFrame:
    if recovered.empty:
        return pd.DataFrame()
    frame = recovered.merge(event_times[["canonical_event_id", "v2_decision_cutoff_utc"]],
                            on="canonical_event_id", how="inner", validate="many_to_one")
    roots = independence_root_map(recovered, event_times)
    base_roots = base_recovered_root_map(recovered, event_times)
    rows = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        published = pd.to_datetime(row.published_at_utc, utc=True, errors="coerce")
        precise = str(row.timestamp_precision) in PRECISE_TIMESTAMPS
        causal = bool(pd.notna(published) and precise and published <= row.v2_decision_cutoff_utc)
        official = bool(row.official_root_accepted) and causal
        independent = bool(row.strict_independent_media_root_accepted) and causal
        publisher = str(row.publisher_canonical)
        official_id = stable_id("OFFICIAL_ROOT_", row.canonical_event_id, publisher) if official else ""
        independent_id = roots.get(row_index, ("", False))[0] if independent else ""
        base_root = base_roots.get(row_index) if independent else None
        if base_root:
            independent_id = base_root[0]
        if independent and not independent_id:
            independent_id = stable_id("MEDIA_ROOT_", row.canonical_event_id, publisher, row.raw_capture_sha256)
        pairwise_merged = bool(
            independent and (roots.get(row_index, ("", False))[1] or base_root)
        )
        if base_root:
            syndication_id = independent_id
        elif bool(row.likely_same_information_root):
            syndication_id = stable_id("SYN_ROOT_", row.canonical_event_id, row.body_char5_jaccard)
        else:
            syndication_id = independent_id or official_id or stable_id("AUX_ROOT_", row.canonical_event_id, row.url)
        acceptance = "CAUSAL_OFFICIAL_ROOT" if official else (
            "CAUSAL_INDEPENDENT_MEDIA_ROOT" if independent else str(row.status)
        )
        rows.append({
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_SOURCE_GRAPH_V1",
            "source_node_id": stable_id("NODE_", row.canonical_event_id, row.url),
            "canonical_event_id": str(row.canonical_event_id),
            "market": str(row.market),
            "ticker": str(row.ticker),
            "article_uid": str(row.candidate_source_uid),
            "source_kind": str(row.source_type),
            "origin_source_id": stable_id("ORIGIN_", publisher),
            "publisher_canonical": publisher,
            "publisher_class": str(row.publisher_class),
            "hosting_domain": str(row.hosting_domain),
            "published_at_utc": published,
            "timestamp_precision": str(row.timestamp_precision),
            "timestamp_confidence": str(row.timestamp_confidence),
            "canonical_url": str(row.canonical_url),
            "syndication_root_id": syndication_id,
            "official_origin_id": official_id,
            "independent_reporting_root_id": independent_id,
            "source_root_type": str(row.source_root_type),
            "pairwise_syndication_merged": pairwise_merged,
            "root_acceptance_status": acceptance,
            "is_causal_v2_t2": causal,
            "counted_causal_information_root": bool(official or independent),
            "counted_independent_media_root": independent,
            "counted_official_root": official,
            "is_major_media": str(row.publisher_class) in MAJOR_CLASSES,
            "is_specialist_media": str(row.publisher_class) in SPECIALIST_CLASSES,
            "event_match_evidence": (
                f"headline_seq={float(row.headline_sequence_ratio):.6f};"
                f"body_seq={float(row.body_sequence_ratio):.6f};"
                f"token_overlap={int(row.salient_token_overlap_count)}"
                + (f";base_root_link={base_root[1]}" if base_root else "")
            ),
            "provenance_evidence": str(row.raw_capture_sha256),
            "outcome_fields_read": "",
        })
    return pd.DataFrame(rows)


def metadata_aux_nodes(event_times: pd.DataFrame, existing_urls: set[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if LOCAL.is_file():
        local = pd.read_parquet(LOCAL)
        for row in local.itertuples(index=False):
            key = (str(row.canonical_event_id), str(row.alternate_url))
            if key in existing_urls:
                continue
            rows.append({
                "canonical_event_id": key[0], "market": str(row.market), "ticker": str(row.ticker),
                "article_uid": str(row.alternate_article_uid), "source_kind": "HEADLINE_METADATA_AUX",
                "publisher_canonical": str(row.alternate_origin_publisher),
                "publisher_class": str(row.alternate_publisher_class),
                "hosting_domain": host_of(row.alternate_url), "published_at_utc": row.alternate_published_at_utc,
                "timestamp_precision": "EXACT_SECOND", "timestamp_confidence": "MEDIUM_RSS_UNVERIFIED_BODY",
                "canonical_url": str(row.alternate_url), "root_acceptance_status": str(row.relation_class),
                "event_match_evidence": str(row.event_match_method), "provenance_evidence": str(row.alternate_source_provider),
            })
    if RSS.is_file():
        rss = pd.read_parquet(RSS)
        for row in rss.itertuples(index=False):
            key = (str(row.canonical_event_id), str(row.direct_url_candidate))
            if key in existing_urls:
                continue
            rows.append({
                "canonical_event_id": key[0], "market": str(row.market), "ticker": str(row.ticker),
                "article_uid": str(row.discovery_uid), "source_kind": "RSS_DISCOVERY_METADATA_AUX",
                "publisher_canonical": str(row.publisher_rss), "publisher_class": str(row.publisher_class),
                "hosting_domain": host_of(row.direct_url_candidate), "published_at_utc": row.published_at_utc,
                "timestamp_precision": str(row.timestamp_precision), "timestamp_confidence": str(row.timestamp_confidence),
                "canonical_url": str(row.direct_url_candidate), "root_acceptance_status": "BODY_NOT_VERIFIED",
                "event_match_evidence": f"headline_seq={float(row.headline_sequence_ratio):.6f}",
                "provenance_evidence": str(row.raw_rss_sha256),
            })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).merge(event_times[["canonical_event_id", "v2_decision_cutoff_utc"]],
                                     on="canonical_event_id", how="inner", validate="many_to_one")
    frame["source_node_id"] = [stable_id("NODE_", row.canonical_event_id, row.canonical_url) for row in frame.itertuples()]
    frame["origin_source_id"] = [stable_id("ORIGIN_", value) for value in frame["publisher_canonical"]]
    frame["syndication_root_id"] = ""
    frame["official_origin_id"] = ""
    frame["independent_reporting_root_id"] = ""
    frame["source_root_type"] = "METADATA_AUX"
    frame["is_causal_v2_t2"] = False
    frame["counted_causal_information_root"] = False
    frame["counted_independent_media_root"] = False
    frame["counted_official_root"] = False
    frame["is_major_media"] = frame["publisher_class"].isin(MAJOR_CLASSES)
    frame["is_specialist_media"] = frame["publisher_class"].isin(SPECIALIST_CLASSES)
    frame["outcome_fields_read"] = ""
    frame["schema_version"] = "HIST_NEWS_MULTISOURCE_V2_SOURCE_GRAPH_V1"
    return frame.drop(columns=["v2_decision_cutoff_utc"])


def build_features(events: pd.DataFrame, event_times: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    base = events.merge(event_times, on=["canonical_event_id", "market", "ticker"], how="inner", validate="one_to_one")
    causal = graph.loc[graph["counted_causal_information_root"].fillna(False).astype(bool)].copy()
    grouped = {key: group for key, group in causal.groupby("canonical_event_id", sort=False)} if not causal.empty else {}
    rows = []
    for event in base.itertuples(index=False):
        group = grouped.get(str(event.canonical_event_id), causal.iloc[0:0])
        media = group.loc[group["counted_independent_media_root"].fillna(False).astype(bool)]
        official = group.loc[group["counted_official_root"].fillna(False).astype(bool)]
        info_root_ids = set(group["official_origin_id"].astype(str)) | set(group["independent_reporting_root_id"].astype(str))
        info_root_ids.discard("")
        media_root_ids = set(media["independent_reporting_root_id"].astype(str))
        media_root_ids.discard("")
        official_ids = set(official["official_origin_id"].astype(str))
        official_ids.discard("")
        published = sorted(pd.to_datetime(group["published_at_utc"], utc=True, errors="coerce").dropna().tolist())
        seconds_to_second = float((published[1] - published[0]).total_seconds()) if len(published) >= 2 else math.nan
        rows.append({
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_CAUSAL_FEATURES_V1",
            "canonical_event_id": str(event.canonical_event_id),
            "market": str(event.market),
            "ticker": str(event.ticker),
            "base_v1_event_time_utc": event.base_v1_event_time_utc,
            "v2_resolved_event_time_utc": event.v2_resolved_event_time_utc,
            "v2_decision_cutoff_utc": event.v2_decision_cutoff_utc,
            "event_time_changed": bool(event.event_time_changed),
            "causal_information_root_count_t2": len(info_root_ids),
            "independent_media_root_count_t2": len(media_root_ids),
            "official_source_count_t2": len(official_ids),
            "unique_origin_publisher_count_t2": int(group["publisher_canonical"].nunique()),
            "unique_publisher_class_count_t2": int(group["publisher_class"].nunique()),
            "major_media_count_t2": int(group["is_major_media"].fillna(False).astype(bool).sum()),
            "biotech_specialist_count_t2": int(group["is_specialist_media"].fillna(False).astype(bool).sum()),
            "issuer_official_present_t2": bool(len(official_ids) > 0),
            "regulatory_present_t2": bool(group["publisher_class"].astype(str).eq("REGULATORY").any()),
            "multisource_exact_ge2_t2": bool(len(info_root_ids) >= 2),
            "independent_media_ge2_t2": bool(len(media_root_ids) >= 2),
            "official_plus_media_confirmed_t2": bool(official_ids and media_root_ids),
            "publisher_entropy_t2": entropy(group["publisher_canonical"].astype(str)),
            "publisher_class_entropy_t2": entropy(group["publisher_class"].astype(str)),
            "seconds_to_second_confirmation": seconds_to_second,
            "outcome_fields_read": "",
        })
    return pd.DataFrame(rows)


def main() -> int:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    event_policy = json.loads(EVENT_TIME_POLICY.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind") or not event_policy.get("outcome_blind"):
        raise RuntimeError("outcome-blind policy guard failed")
    queue = pd.read_parquet(QUEUE)[["canonical_event_id", "market", "ticker", "issuer_name", "event_time_utc"]].copy()
    queue = queue.rename(columns={"issuer_name": "issuer"})
    queue["event_time_utc"] = pd.to_datetime(queue["event_time_utc"], utc=True, errors="raise")
    recovered_parts = []
    if RECOVERED.is_file():
        recovered_parts.append(pd.read_parquet(RECOVERED))
    if OFFICIAL_RECOVERED.is_file():
        recovered_parts.append(pd.read_parquet(OFFICIAL_RECOVERED))
    if KIND_RECOVERED.is_file():
        recovered_parts.append(pd.read_parquet(KIND_RECOVERED))
    recovered = pd.concat(recovered_parts, ignore_index=True, sort=False) if recovered_parts else pd.DataFrame()

    event_times = build_event_time_manifest(queue, recovered)
    current = current_nodes(queue, event_times)
    current["pairwise_syndication_merged"] = False
    additional = recovered_nodes(recovered, event_times)
    parts = [current]
    if not additional.empty:
        parts.append(additional)
    existing_urls = set(zip(additional["canonical_event_id"].astype(str), additional["canonical_url"].astype(str))) if not additional.empty else set()
    auxiliary = metadata_aux_nodes(event_times, existing_urls)
    if not auxiliary.empty:
        parts.append(auxiliary)
    graph = pd.concat(parts, ignore_index=True, sort=False)
    graph = graph.sort_values(["canonical_event_id", "source_node_id"], kind="mergesort").drop_duplicates(
        ["canonical_event_id", "source_node_id"], keep="first"
    ).reset_index(drop=True)
    features = build_features(queue, event_times, graph)
    syndication = graph[[
        "canonical_event_id", "source_node_id", "article_uid", "publisher_canonical",
        "syndication_root_id", "independent_reporting_root_id", "source_root_type",
        "pairwise_syndication_merged", "root_acceptance_status", "outcome_fields_read",
    ]].copy()
    atomic_parquet(EVENT_TIME_MANIFEST, event_times)
    atomic_parquet(SOURCE_GRAPH, graph)
    atomic_parquet(SYNDICATION_ROOTS, syndication)
    atomic_parquet(CAUSAL_FEATURES, features)

    changed = features.loc[features["event_time_changed"]]
    multi = features.loc[features["multisource_exact_ge2_t2"]]
    media_multi = features.loc[features["independent_media_ge2_t2"]]
    official_media = features.loc[features["official_plus_media_confirmed_t2"]]
    if multi.empty:
        freeze_blocker = (
            "No strict reconstructed multi-source exact events remain after event-time, "
            "body, attribution, and syndication QA; continue outcome-blind collection "
            "or conclude that a trainable V2 corpus is unavailable."
        )
    else:
        freeze_blocker = (
            "Only a tiny number of reconstructed multi-source exact events; continue "
            "collection/source-family audit before any freeze or model."
        )
    report = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_SOURCE_GRAPH_AUDIT_V1",
        "generated_at_utc": utc_now(),
        "outcome_blind": True,
        "outcome_fields_read": [],
        "events": int(len(features)),
        "source_nodes": int(len(graph)),
        "counted_causal_root_nodes": int(graph["counted_causal_information_root"].fillna(False).astype(bool).sum()),
        "events_with_v2_event_time_change": int(len(changed)),
        "multisource_exact_ge2_events": int(len(multi)),
        "independent_media_ge2_events": int(len(media_multi)),
        "official_plus_media_confirmed_events": int(len(official_media)),
        "market_multisource_counts": {
            str(key): int(value) for key, value in multi["market"].value_counts().items()
        },
        "headline_metadata_aux_nodes": int(graph["source_root_type"].astype(str).eq("METADATA_AUX").sum()),
        "pairwise_syndication_merged_nodes": int(graph["pairwise_syndication_merged"].eq(True).sum()),
        "duplicate_source_node_ids": int(graph.duplicated(["canonical_event_id", "source_node_id"]).sum()),
        "future_article_leakage": int((
            graph["counted_causal_information_root"].fillna(False).astype(bool)
            & ~graph["is_causal_v2_t2"].fillna(False).astype(bool)
        ).sum()),
        "source_freeze_ready": False,
        "source_freeze_blocker": freeze_blocker,
        "model_training_allowed": False,
        "base_v1_mutated": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "artifacts": {
            EVENT_TIME_MANIFEST.name: sha256_file(EVENT_TIME_MANIFEST),
            SOURCE_GRAPH.name: sha256_file(SOURCE_GRAPH),
            SYNDICATION_ROOTS.name: sha256_file(SYNDICATION_ROOTS),
            CAUSAL_FEATURES.name: sha256_file(CAUSAL_FEATURES),
        },
    }
    atomic_json(REPORT, report)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update({
        "updated_at_utc": utc_now(),
        "status": "SOURCE_GRAPH_RECONSTRUCTED_COLLECTION_CONTINUES",
        "phase": "OUTCOME_BLIND_MULTISOURCE_INFORMATION_GAIN",
        "v2_event_time_changes": report["events_with_v2_event_time_change"],
        "v2_multisource_exact_ge2_events": report["multisource_exact_ge2_events"],
        "v2_independent_media_ge2_events": report["independent_media_ge2_events"],
        "v2_official_plus_media_events": report["official_plus_media_confirmed_events"],
        "labels_read": False,
        "source_frozen": False,
        "split_frozen": False,
        "model_training_allowed": False,
        "base_v1_membership_mutable": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
    })
    atomic_json(STATE, state)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
