from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import pandas as pd
from bs4 import BeautifulSoup

from historical_news_article_backfill import (
    PublicFetcher,
    RequestRegistry,
    parse_html_article,
    raw_store,
)


ROOT = Path(__file__).resolve().parent
POLICY_MANIFEST = ROOT / "research" / "multisource_v2" / "POLICY_FREEZE_MANIFEST.json"
SEEDS = ROOT / "research" / "multisource_v2" / "PUBLIC_SOURCE_SEEDS.json"
QUEUE = ROOT / "research" / "hist_news_multisource_v2" / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
CURRENT = WORKING / "V2_CURRENT_SOURCE_PROVENANCE.parquet"
RSS = WORKING / "V2_DISCOVERED_RSS_ITEMS.parquet"
GDELT_GAL = WORKING / "V2_DISCOVERED_GDELT_GAL_ITEMS.parquet"
LOCAL = WORKING / "V2_LOCAL_CACHE_LINK_AUDIT.parquet"
BASE_ARTICLES = ROOT / "data" / "HIST_HIGH_CONF_NEWS_BASE_V1" / "ARTICLES_NORMALIZED.parquet"
RECOVERED = WORKING / "V2_RECOVERED_PUBLIC_ARTICLES.parquet"
FETCH_AUDIT = WORKING / "V2_BODY_FETCH_AUDIT.parquet"
REPORT = ROOT / "research" / "multisource_v2" / "BODY_RECOVERY_REPORT.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"

AGGREGATOR_DOMAINS = {
    "bing.com", "google.com", "news.google.com", "msn.com", "naver.com",
    "news.naver.com", "n.news.naver.com", "v.daum.net", "yahoo.com",
}

HIGH_ORIGIN_TIMESTAMP_CONFIDENCE = {"HIGH", "HIGH_ORIGIN_PAGE"}
OFFICIAL_TIMESTAMP_CONFIDENCE = HIGH_ORIGIN_TIMESTAMP_CONFIDENCE | {"MEDIUM_DETERMINISTIC_IDENTIFIER"}

DOMAIN_PUBLISHERS: dict[str, tuple[str, str]] = {
    "jw-pharma.co.kr": ("JW중외제약", "ISSUER_OFFICIAL"),
    "gccorp.com": ("GC녹십자", "ISSUER_OFFICIAL"),
    "newspim.com": ("뉴스핌", "MAJOR_FINANCIAL"),
    "supple.kr": ("SBS Biz", "MAJOR_FINANCIAL"),
    "newstap.co.kr": ("뉴스탭", "LOCAL_OTHER_MEDIA"),
    "etnews.com": ("전자신문", "MAJOR_FINANCIAL"),
    "edaily.co.kr": ("이데일리", "MAJOR_FINANCIAL"),
    "fnnews.com": ("파이낸셜뉴스", "MAJOR_FINANCIAL"),
    "mt.co.kr": ("머니투데이", "MAJOR_FINANCIAL"),
    "etoday.co.kr": ("이투데이", "MAJOR_FINANCIAL"),
    "docdocdoc.co.kr": ("청년의사", "PHARMA_SPECIALIST"),
    "segye.com": ("세계일보", "MAJOR_GENERAL"),
    "mbnmoney.mbn.co.kr": ("MBN", "MAJOR_GENERAL"),
    "sportalkorea.com": ("스포탈코리아", "LOCAL_OTHER_MEDIA"),
    "kyeonggi.com": ("경기일보", "LOCAL_OTHER_MEDIA"),
    "topdaily.kr": ("톱데일리", "MAJOR_FINANCIAL"),
    "hankyung.com": ("한국경제", "MAJOR_FINANCIAL"),
    "joongang.co.kr": ("중앙일보", "MAJOR_GENERAL"),
    "inews24.com": ("아이뉴스24", "MAJOR_GENERAL"),
    "bokuennews.com": ("보건신문", "PHARMA_SPECIALIST"),
    "yakup.com": ("약업신문", "PHARMA_SPECIALIST"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def normalized_text(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", " ", str(value or "").casefold()).strip()


def char_ngrams(value: Any, n: int = 5) -> set[str]:
    text = normalized_text(value).replace(" ", "")
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[index:index + n] for index in range(len(text) - n + 1)}


def text_similarity(left: Any, right: Any) -> tuple[float, float]:
    a = normalized_text(left)
    b = normalized_text(right)
    sequence = SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0
    ga, gb = char_ngrams(a), char_ngrams(b)
    jaccard = len(ga & gb) / len(ga | gb) if ga and gb else 0.0
    return float(sequence), float(jaccard)


def salient_token_overlap(reference: Any, candidate_text: Any) -> tuple[int, float]:
    stop = {
        "국내", "미국", "관련", "공개", "발표", "신청", "개최", "종합", "단독",
        "대한", "통해", "위한", "에서", "으로", "한다", "했다", "밝혔다",
    }
    def token_key(token: str) -> str:
        return re.sub(r"(?:에서는|으로는|에게서|부터|까지|으로|에서|에게|한테|처럼|보다|은|는|이|가|을|를|의|과|와|도)$", "", token)

    reference_tokens = {
        token_key(token) for token in normalized_text(reference).split()
        if len(token) >= 2 and token not in stop and not token.isdigit()
    }
    reference_tokens.discard("")
    candidate_tokens = {token_key(token) for token in normalized_text(candidate_text).split()}
    overlap = reference_tokens & candidate_tokens
    ratio = len(overlap) / len(reference_tokens) if reference_tokens else 0.0
    return len(overlap), float(ratio)


def nonarticle_body_reason(body: Any) -> str:
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    if 160 <= len(text) < 800:
        midpoint = len(text) // 2
        repeated_half = SequenceMatcher(None, text[:midpoint], text[midpoint:]).ratio()
        if repeated_half >= 0.90:
            return "SHORT_REPEATED_CAPTION"
    lowered = text.casefold()
    if len(text) < 600 and (
        ("사진=" in text or "사진 =" in text or "photo=" in lowered)
        and ("재판매" in text or "db 금지" in lowered or "provided" in lowered)
    ):
        return "SHORT_PHOTO_CAPTION"
    return ""


def attribution_anchor_consistent(reference_headline: Any, candidate_text: Any) -> tuple[bool, str]:
    headline = str(reference_headline or "")
    candidate = normalized_text(candidate_text)
    anchors = re.findall(r"([가-힣A-Za-z0-9]{2,16})(?:증권|證|리서치)", headline)
    aliases = {
        "한투": {"한투", "한국투자"},
        "한화투자": {"한화투자", "한화"},
        "NH투자": {"nh투자", "nh"},
        "KB": {"kb"},
    }
    for anchor in anchors:
        variants = aliases.get(anchor, {normalized_text(anchor)})
        if not any(normalized_text(value) in candidate for value in variants):
            return False, anchor
    return True, ""


def english_distinctive_anchor_consistent(reference_headline: Any, candidate_text: Any) -> tuple[bool, str]:
    """Require a named/distinctive English event anchor, not generic biotech words."""
    reference = str(reference_headline or "")
    if len(re.findall(r"[A-Za-z]", reference)) < 8:
        return True, ""
    reference = re.sub(r"\b([A-Z])\.([A-Z])\.", r"\1\2", reference)
    candidate = re.sub(r"\b([A-Z])\.([A-Z])\.", r"\1\2", str(candidate_text or ""))
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "in", "into", "is", "it", "its", "of", "on", "or", "says", "said", "the", "to",
        "with", "after", "before", "first", "major", "new", "more", "could", "will", "can",
        "company", "companies", "reports", "reported", "announces", "announced", "begins",
        "covid", "covid19", "coronavirus", "pandemic", "vaccine", "vaccines", "vaccination",
        "vaccinations", "drug", "drugs", "treatment", "trial", "trials", "phase", "results",
        "result", "study", "studies", "health", "patients", "patient", "update", "news",
    }

    def normalized_token(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold().removesuffix("'s"))

    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9'.-]*", reference)
    named = []
    fallback = []
    for raw in raw_tokens:
        token = normalized_token(raw)
        if len(token) < 2 or token in stop:
            continue
        fallback.append(token)
        if raw[0].isupper() or raw.isupper() or any(character.isdigit() for character in raw):
            named.append(token)
    anchors = list(dict.fromkeys(named or fallback))
    if not anchors:
        return True, ""
    candidate_tokens = {
        normalized_token(value) for value in re.findall(r"[A-Za-z][A-Za-z0-9'.-]*", candidate)
    }
    candidate_tokens.discard("")

    def present(anchor: str) -> bool:
        if anchor == "us":
            return "us" in candidate_tokens or {"united", "states"}.issubset(candidate_tokens)
        if anchor == "uk":
            return "uk" in candidate_tokens or {"united", "kingdom"}.issubset(candidate_tokens)
        aliases = {
            "germany": {"germany", "german"},
            "korea": {"korea", "korean"},
            "japan": {"japan", "japanese"},
            "china": {"china", "chinese"},
        }
        if anchor == "eu":
            return "eu" in candidate_tokens or {"european", "union"}.issubset(candidate_tokens)
        if anchor in aliases:
            return bool(aliases[anchor] & candidate_tokens)
        return anchor in candidate_tokens

    matched = [anchor for anchor in anchors if present(anchor)]
    required = min(2, len(anchors)) if named else min(2, max(1, math.ceil(len(anchors) * 0.34)))
    if len(matched) < required:
        missing = [anchor for anchor in anchors if anchor not in matched]
        return False, ",".join(missing[:4])
    return True, ""


def host_of(url: Any) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def domain_is(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def resolve_publisher(url: str, parsed_publisher: Any, hint: Any, class_hint: Any) -> tuple[str, str, str]:
    host = host_of(url)
    for domain, (publisher, publisher_class) in DOMAIN_PUBLISHERS.items():
        if domain_is(host, domain):
            return publisher, publisher_class, "DIRECT_DOMAIN_REGISTRY"
    parsed = str(parsed_publisher or "").strip()
    if parsed and parsed.casefold() not in {"msn", "bing", "google news", "naver news"}:
        return parsed[:160], str(class_hint or "LOCAL_OTHER_MEDIA"), "ORIGIN_PAGE_METADATA"
    publisher_hint = str(hint or "UNKNOWN").strip()
    if publisher_hint and publisher_hint != "UNKNOWN":
        return publisher_hint[:160], str(class_hint or "LOCAL_OTHER_MEDIA"), "DISCOVERY_METADATA_HINT"
    return "UNKNOWN", "AGGREGATOR", "UNRESOLVED"


def enrich_page_fields(raw: bytes, url: str, parsed: dict[str, Any]) -> dict[str, Any]:
    """Recover publisher-specific fields without weakening timestamp provenance."""
    fields = dict(parsed)
    soup = BeautifulSoup(raw, "html.parser")
    host = host_of(url)

    def node_text(selector: str) -> str:
        node = soup.select_one(selector)
        return node.get_text(" ", strip=True) if node else ""

    if domain_is(host, "newstap.co.kr"):
        fields["body"] = node_text("#article-view-content-div") or node_text(".article-body") or fields.get("body", "")
    elif domain_is(host, "jw-pharma.co.kr"):
        fields["headline"] = node_text(".detail_top h3") or fields.get("headline", "")
        fields["body"] = node_text(".detail_cont") or fields.get("body", "")
        match = re.search(r"contentsCd=(\d{12})", url)
        if not match:
            hidden = soup.select_one('input[name="contentsCd"]')
            match = re.match(r"(\d{12})", str(hidden.get("value") if hidden else ""))
        if match:
            local = datetime.strptime(match.group(1), "%y%m%d%H%M%S").replace(tzinfo=timezone(timedelta(hours=9)))
            fields.update({
                "published_at": local.astimezone(timezone.utc),
                "published_at_source": "PUBLIC_ISSUER_CONTENT_ID_YYMMDDHHMMSS",
                "published_at_precision": "EXACT_SECOND",
                "published_at_confidence": "MEDIUM_DETERMINISTIC_IDENTIFIER",
                "timezone_inferred": False,
            })
    elif domain_is(host, "gccorp.com"):
        fields["headline"] = node_text(".view-tit .tit") or fields.get("headline", "")
        fields["body"] = node_text(".view-content") or fields.get("body", "")
        date_text = node_text(".view-tit .date")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
            fields.update({
                "published_at": pd.Timestamp(date_text, tz="Asia/Seoul").tz_convert("UTC"),
                "published_at_source": "PUBLIC_ISSUER_PAGE_DATE_ONLY",
                "published_at_precision": "DATE_ONLY",
                "published_at_confidence": "LOW_NON_CAUSAL_DATE_ONLY",
                "timezone_inferred": False,
            })
    elif domain_is(host, "bokuennews.com"):
        visible = soup.get_text(" ", strip=True)
        match = re.search(r"(20\d{2})[.-](\d{2})[.-](\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?", visible)
        if match:
            values = [int(value or 0) for value in match.groups()]
            local = datetime(*values).replace(tzinfo=timezone(timedelta(hours=9)))
            fields.update({
                "published_at": local.astimezone(timezone.utc),
                "published_at_source": "PUBLIC_ORIGIN_VISIBLE_BYLINE_TIMESTAMP",
                "published_at_precision": "EXACT_SECOND" if match.group(6) else "EXACT_MINUTE",
                "published_at_confidence": "HIGH",
                "timezone_inferred": False,
            })
    elif domain_is(host, "kyeonggi.com"):
        visible = soup.get_text(" ", strip=True)
        match = re.search(r"승인\s+(20\d{2})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", visible)
        if match:
            local = datetime(*(int(value) for value in match.groups())).replace(
                tzinfo=timezone(timedelta(hours=9))
            )
            fields.update({
                "published_at": local.astimezone(timezone.utc),
                "published_at_source": "PUBLIC_ORIGIN_VISIBLE_APPROVAL_TIMESTAMP",
                "published_at_precision": "EXACT_MINUTE",
                "published_at_confidence": "HIGH",
                "timezone_inferred": False,
            })
    elif domain_is(host, "yakup.com"):
        visible = soup.get_text(" ", strip=True)
        match = re.search(r"입력\s+(20\d{2})\.(\d{2})\.(\d{2})\s+(\d{2}):(\d{2})", visible)
        if match:
            local = datetime(*(int(value) for value in match.groups())).replace(
                tzinfo=timezone(timedelta(hours=9))
            )
            fields.update({
                "published_at": local.astimezone(timezone.utc),
                "published_at_source": "PUBLIC_ORIGIN_VISIBLE_INPUT_TIMESTAMP",
                "published_at_precision": "EXACT_MINUTE",
                "published_at_confidence": "HIGH",
                "timezone_inferred": False,
            })
    return fields


def prepare_candidates() -> pd.DataFrame:
    queue = pd.read_parquet(QUEUE)[["canonical_event_id", "market", "ticker", "issuer_name", "event_time_utc"]].copy()
    queue = queue.rename(columns={"issuer_name": "issuer"})
    queue["event_time_utc"] = pd.to_datetime(queue["event_time_utc"], utc=True, errors="raise")
    current = pd.read_parquet(CURRENT)[[
        "canonical_event_id", "current_article_uid", "origin_publisher_resolved", "current_headline"
    ]].copy()
    event = queue.merge(current, on="canonical_event_id", how="inner", validate="one_to_one")
    parts: list[pd.DataFrame] = []

    if RSS.is_file():
        rss = pd.read_parquet(RSS)
        rss = rss.loc[
            rss["causal_rss_candidate_t2"].fillna(False).astype(bool)
            & rss["origin_distinct_from_current"].fillna(False).astype(bool)
        ].copy()
        if not rss.empty:
            rss_publisher_hint = (
                rss["origin_publisher_candidate"].astype(str)
                if "origin_publisher_candidate" in rss.columns
                else rss["publisher_rss"].astype(str)
            )
            parts.append(pd.DataFrame({
                "canonical_event_id": rss["canonical_event_id"].astype(str),
                "url": rss["direct_url_candidate"].astype(str),
                "publisher_hint": rss_publisher_hint,
                "publisher_class_hint": rss["publisher_class"].astype(str),
                "source_type": "PUBLIC_RSS_DIRECT_RECOVERY",
                "discovery_timestamp_utc": rss["published_at_utc"],
                "discovery_timestamp_provenance": rss["timestamp_provenance"].astype(str),
                "candidate_headline": rss["headline"].astype(str),
                "candidate_source_uid": rss["discovery_uid"].astype(str),
                "discovery_evidence": "Outcome-blind public RSS semantic/time candidate",
            }))

    if GDELT_GAL.is_file():
        gal = pd.read_parquet(GDELT_GAL)
        gal = gal.loc[gal["origin_distinct_from_current"].fillna(False).astype(bool)].copy()
        already_verified_ids: set[str] = set()
        if RECOVERED.is_file():
            prior_verified = pd.read_parquet(RECOVERED)
            already_verified_ids = set(prior_verified.loc[
                prior_verified["strict_independent_media_root_accepted"].fillna(False).astype(bool)
                | prior_verified["official_root_accepted"].fillna(False).astype(bool),
                "candidate_source_uid",
            ].astype(str))
        description = (
            gal["description"].fillna("").astype(str)
            if "description" in gal.columns else pd.Series("", index=gal.index, dtype="object")
        )
        anchor_ok = pd.Series([
            english_distinctive_anchor_consistent(reference, f"{headline} {context}")[0]
            for reference, headline, context in zip(
                gal["current_headline"].astype(str), gal["headline"].astype(str), description
            )
        ], index=gal.index, dtype="bool")
        gal = gal.loc[
            anchor_ok | gal["discovery_uid"].astype(str).isin(already_verified_ids)
        ].copy()
        if not gal.empty:
            parts.append(pd.DataFrame({
                "canonical_event_id": gal["canonical_event_id"].astype(str),
                "url": gal["direct_url_candidate"].astype(str),
                "publisher_hint": gal["publisher_candidate"].astype(str),
                "publisher_class_hint": gal["publisher_class"].astype(str),
                "source_type": "GDELT_GAL_DIRECT_RECOVERY",
                "discovery_timestamp_utc": gal["gal_seen_at_utc"],
                "discovery_timestamp_provenance": gal["timestamp_provenance"].astype(str),
                "candidate_headline": gal["headline"].astype(str),
                "candidate_source_uid": gal["discovery_uid"].astype(str),
                "discovery_evidence": (
                    "Outcome-blind GDELT GAL URL discovery; GAL seen time is not publication time"
                ),
            }))

    if LOCAL.is_file():
        local = pd.read_parquet(LOCAL)
        local = local.loc[
            local["relation_class"].eq("HEADLINE_ONLY_DISTINCT_MEDIA_CANDIDATE")
            & local["causal_published_by_t2"].fillna(False).astype(bool)
        ].copy()
        if not local.empty:
            parts.append(pd.DataFrame({
                "canonical_event_id": local["canonical_event_id"].astype(str),
                "url": local["alternate_url"].astype(str),
                "publisher_hint": local["alternate_origin_publisher"].astype(str),
                "publisher_class_hint": local["alternate_publisher_class"].astype(str),
                "source_type": "LOCAL_RSS_DIRECT_RECOVERY",
                "discovery_timestamp_utc": local["alternate_published_at_utc"],
                "discovery_timestamp_provenance": "EXISTING_PUBLIC_RSS_PUBDATE",
                "candidate_headline": local["alternate_headline"].astype(str),
                "candidate_source_uid": local["alternate_article_uid"].astype(str),
                "discovery_evidence": "Outcome-blind cached RSS high semantic/time match",
            }))

    seed_payload = json.loads(SEEDS.read_text(encoding="utf-8"))
    if seed_payload.get("outcome_fields_read") != []:
        raise RuntimeError("seed outcome guard failed")
    seed_rows = []
    for index, seed in enumerate(seed_payload.get("seeds", [])):
        seed_rows.append({
            **seed,
            "discovery_timestamp_utc": pd.NaT,
            "discovery_timestamp_provenance": "",
            "candidate_headline": "",
            "candidate_source_uid": f"PUBLIC_SEED_{index:04d}",
        })
    if seed_rows:
        parts.append(pd.DataFrame(seed_rows))

    if not parts:
        return event.iloc[0:0].copy()
    candidates = pd.concat(parts, ignore_index=True, sort=False)
    candidates = candidates.merge(event, on="canonical_event_id", how="inner", validate="many_to_one")
    candidates["event_time_utc"] = pd.to_datetime(candidates["event_time_utc"], utc=True, errors="raise")
    candidates["decision_cutoff_utc"] = candidates["event_time_utc"] + pd.Timedelta(minutes=2)
    candidates["discovery_timestamp_utc"] = pd.to_datetime(candidates["discovery_timestamp_utc"], utc=True, errors="coerce")
    candidates = candidates.loc[candidates["url"].astype(str).str.startswith(("http://", "https://"))].copy()
    candidates = candidates.sort_values(["canonical_event_id", "url", "candidate_source_uid"], kind="mergesort")
    return candidates.drop_duplicates(["canonical_event_id", "url"], keep="first").reset_index(drop=True)


def base_article_lookup(candidates: pd.DataFrame) -> dict[str, dict[str, str]]:
    wanted = set(candidates["current_article_uid"].astype(str))
    base = pd.read_parquet(BASE_ARTICLES, columns=["article_uid", "headline", "body"])
    base = base.loc[base["article_uid"].astype(str).isin(wanted)]
    return {
        str(row.article_uid): {"headline": str(row.headline or ""), "body": str(row.body or "")}
        for row in base.itertuples(index=False)
    }


def choose_timestamp(parsed: dict[str, Any], candidate: Any) -> tuple[pd.Timestamp | None, str, str, str]:
    parsed_stamp = pd.to_datetime(parsed.get("published_at"), utc=True, errors="coerce")
    precision = str(parsed.get("published_at_precision") or "UNKNOWN")
    if pd.notna(parsed_stamp) and precision in {"EXACT_SECOND", "EXACT_MINUTE"}:
        confidence = str(parsed.get("published_at_confidence") or "HIGH_ORIGIN_PAGE")
        return parsed_stamp, str(parsed.get("published_at_source") or "ORIGIN_PAGE"), precision, confidence
    rss_stamp = pd.to_datetime(candidate.discovery_timestamp_utc, utc=True, errors="coerce")
    if pd.notna(rss_stamp):
        provenance = str(candidate.discovery_timestamp_provenance or "PUBLIC_RSS_PUBDATE")
        confidence = (
            "LOW_GDELT_SEEN_TIME_NOT_PUBLICATION"
            if provenance == "GDELT_GAL_SEEN_TIME_DISCOVERY_ONLY"
            else "MEDIUM_RSS_DIRECT_BODY_VERIFIED"
        )
        return rss_stamp, provenance, "EXACT_SECOND", confidence
    if pd.notna(parsed_stamp):
        return parsed_stamp, str(parsed.get("published_at_source") or "ORIGIN_PAGE"), precision, "LOW_NON_MINUTE"
    return None, "UNKNOWN", "UNKNOWN", "UNKNOWN"


def recovery_status(row: dict[str, Any]) -> str:
    if not row["fetch_succeeded"]:
        return "FETCH_NOT_AVAILABLE"
    if not row.get("english_distinctive_anchor_consistent", True):
        return "EVENT_DISTINCTIVE_ANCHOR_MISMATCH"
    if not row["event_match_high"]:
        return "EVENT_MATCH_REJECTED"
    if row["body_length"] < 80:
        return "BODY_TOO_SHORT"
    if not row.get("body_event_match_high", True):
        return "BODY_EVENT_MATCH_REJECTED"
    if not row.get("body_content_usable", True):
        return "BODY_CONTENT_NOT_ARTICLE"
    if not row.get("attribution_anchor_consistent", True):
        return "EVENT_ATTRIBUTION_ANCHOR_MISMATCH"
    if not row["origin_distinct_from_current"]:
        return "SAME_ORIGIN_OR_UNRESOLVED"
    if not row["published_by_t2"]:
        return "RETROSPECTIVE_AUX_POST_T2"
    if not row.get("timestamp_acceptable_for_root", True):
        return "TIMESTAMP_CONFIDENCE_INSUFFICIENT"
    if row["source_root_type"] == "OFFICIAL_ROOT":
        return "CAUSAL_OFFICIAL_ROOT_VERIFIED"
    if row["likely_same_information_root"]:
        return "CAUSAL_DISTINCT_PUBLISHER_SYNDICATION_ROOT"
    return "CAUSAL_INDEPENDENT_MEDIA_ROOT_VERIFIED"


def run(max_requests: int) -> dict[str, Any]:
    policy = json.loads(POLICY_MANIFEST.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind") or policy.get("label_join_allowed"):
        raise RuntimeError("collection policy guard failed")
    candidates = prepare_candidates()
    base = base_article_lookup(candidates)
    output_root = WORKING / "public_body_recovery"
    prior_recovered: dict[tuple[str, str], Any] = {}
    prior_recovered_by_url: dict[str, Any] = {}
    if RECOVERED.is_file():
        prior_frame = pd.read_parquet(RECOVERED)
        prior_recovered = {
            (str(row.canonical_event_id), str(row.url)): row for row in prior_frame.itertuples(index=False)
        }
        for row in prior_frame.sort_values("captured_at_utc", kind="mergesort").itertuples(index=False):
            if str(getattr(row, "raw_capture_path", "")):
                prior_recovered_by_url[str(row.url)] = row
    registry = RequestRegistry(output_root)
    fetcher = PublicFetcher(output_root, registry, max_requests=max_requests)
    recovered_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    cross_event_raw_reuses = 0

    for candidate in candidates.itertuples(index=False):
        fetched = None
        prior = prior_recovered.get((str(candidate.canonical_event_id), str(candidate.url)))
        if prior is None:
            prior = prior_recovered_by_url.get(str(candidate.url))
            if prior is not None:
                cross_event_raw_reuses += 1
        if prior is not None and str(getattr(prior, "raw_capture_path", "")):
            raw_path = output_root / str(prior.raw_capture_path)
            if raw_path.is_file():
                with gzip.open(raw_path, "rb") as handle:
                    fetched = (handle.read(), str(getattr(prior, "raw_mime_type", "text/html")))
        if fetched is None:
            fetched = fetcher.article(str(candidate.url))
        if fetched is None:
            audit_rows.append({
                "canonical_event_id": str(candidate.canonical_event_id),
                "url": str(candidate.url),
                "candidate_source_uid": str(candidate.candidate_source_uid),
                "source_type": str(candidate.source_type),
                "fetch_succeeded": False,
                "status": "FETCH_NOT_AVAILABLE",
                "outcome_fields_read": "",
            })
            continue
        raw, mime = fetched
        parsed = parse_html_article(raw, str(candidate.url), str(candidate.source_type))
        parsed = enrich_page_fields(raw, str(candidate.url), parsed)
        headline = str(parsed.get("headline") or candidate.candidate_headline or "")
        body = str(parsed.get("body") or "")
        origin_url = str(parsed.get("canonical_url") or candidate.url)
        publisher, publisher_class, publisher_method = resolve_publisher(
            origin_url, parsed.get("publisher_raw"), candidate.publisher_hint, candidate.publisher_class_hint,
        )
        published, timestamp_provenance, timestamp_precision, timestamp_confidence = choose_timestamp(parsed, candidate)
        prior = base.get(str(candidate.current_article_uid), {"headline": str(candidate.current_headline), "body": ""})
        headline_sequence, headline_jaccard = text_similarity(headline, prior["headline"])
        body_sequence, body_jaccard = text_similarity(body, prior["body"])
        token_overlap_count, token_overlap_ratio = salient_token_overlap(
            prior["headline"], headline + " " + body[:2500]
        )
        body_token_overlap_count, body_token_overlap_ratio = salient_token_overlap(
            prior["headline"], body[:4000]
        )
        body_event_match_high = bool(
            body_sequence >= 0.20 and body_jaccard >= 0.05
            or body_token_overlap_count >= 3 and body_token_overlap_ratio >= 0.35
        )
        body_nonarticle_reason = nonarticle_body_reason(body)
        body_content_usable = not body_nonarticle_reason
        attribution_consistent, missing_attribution_anchor = attribution_anchor_consistent(
            prior["headline"], headline + " " + body[:5000]
        )
        english_anchor_consistent, missing_english_anchor = english_distinctive_anchor_consistent(
            prior["headline"], headline + " " + body[:800]
        )
        body_event_match_high = bool(body_event_match_high and english_anchor_consistent)
        event_match_high = bool(
            english_anchor_consistent and (
                headline_sequence >= 0.40 and headline_jaccard >= 0.10
                or body_sequence >= 0.35 and body_jaccard >= 0.08
                or token_overlap_count >= 3 and token_overlap_ratio >= 0.35
            )
        )
        host = host_of(origin_url)
        origin_distinct = bool(
            publisher not in {"", "UNKNOWN", str(candidate.origin_publisher_resolved)}
            and not any(domain_is(host, domain) for domain in AGGREGATOR_DOMAINS)
        )
        is_official = str(candidate.source_type) == "ISSUER_OFFICIAL" or publisher_class == "ISSUER_OFFICIAL"
        likely_same_root = bool(
            not is_official
            and len(body) >= 80
            and (body_sequence >= 0.82 or body_jaccard >= 0.70)
        )
        published_by_t2 = bool(
            pd.notna(published)
            and timestamp_precision in {"EXACT_SECOND", "EXACT_MINUTE"}
            and published <= candidate.decision_cutoff_utc
        )
        source_root_type = "OFFICIAL_ROOT" if is_official else (
            "SYNDICATION_OR_COMMON_INFORMATION_ROOT" if likely_same_root else "MEDIA_REPORTING_ROOT"
        )
        timestamp_acceptable_for_root = bool(
            timestamp_confidence in (
                OFFICIAL_TIMESTAMP_CONFIDENCE if is_official else HIGH_ORIGIN_TIMESTAMP_CONFIDENCE
            )
        )
        raw_rel, raw_sha = raw_store(output_root, raw, "html")
        row: dict[str, Any] = {
            "schema_version": "HIST_NEWS_MULTISOURCE_V2_RECOVERED_ARTICLE_V1",
            "canonical_event_id": str(candidate.canonical_event_id),
            "market": str(candidate.market),
            "ticker": str(candidate.ticker),
            "issuer": str(candidate.issuer),
            "event_time_utc": candidate.event_time_utc,
            "decision_cutoff_utc": candidate.decision_cutoff_utc,
            "current_article_uid": str(candidate.current_article_uid),
            "current_origin_publisher": str(candidate.origin_publisher_resolved),
            "candidate_source_uid": str(candidate.candidate_source_uid),
            "source_type": str(candidate.source_type),
            "url": str(candidate.url),
            "canonical_url": origin_url,
            "hosting_domain": host,
            "publisher_canonical": publisher,
            "publisher_class": publisher_class,
            "publisher_resolution_method": publisher_method,
            "published_at_utc": published,
            "timestamp_provenance": timestamp_provenance,
            "timestamp_precision": timestamp_precision,
            "timestamp_confidence": timestamp_confidence,
            "headline": headline,
            "body": body,
            "body_length": int(len(body)),
            "headline_sequence_ratio": headline_sequence,
            "headline_char5_jaccard": headline_jaccard,
            "body_sequence_ratio": body_sequence,
            "body_char5_jaccard": body_jaccard,
            "salient_token_overlap_count": token_overlap_count,
            "salient_token_overlap_ratio": token_overlap_ratio,
            "body_salient_token_overlap_count": body_token_overlap_count,
            "body_salient_token_overlap_ratio": body_token_overlap_ratio,
            "body_event_match_high": body_event_match_high,
            "body_content_usable": body_content_usable,
            "body_nonarticle_reason": body_nonarticle_reason,
            "attribution_anchor_consistent": attribution_consistent,
            "missing_attribution_anchor": missing_attribution_anchor,
            "english_distinctive_anchor_consistent": english_anchor_consistent,
            "missing_english_distinctive_anchor": missing_english_anchor,
            "event_match_high": event_match_high,
            "origin_distinct_from_current": origin_distinct,
            "published_by_t2": published_by_t2,
            "timestamp_acceptable_for_root": timestamp_acceptable_for_root,
            "likely_same_information_root": likely_same_root,
            "source_root_type": source_root_type,
            "official_root_accepted": bool(
                is_official and event_match_high and body_event_match_high and body_content_usable
                and attribution_consistent
                and len(body) >= 80 and published_by_t2
                and timestamp_acceptable_for_root
            ),
            "strict_independent_media_root_accepted": bool(
                not is_official and event_match_high and body_event_match_high and body_content_usable
                and attribution_consistent
                and len(body) >= 80 and origin_distinct
                and published_by_t2 and timestamp_acceptable_for_root and not likely_same_root
            ),
            "raw_capture_path": raw_rel,
            "raw_capture_sha256": raw_sha,
            "raw_mime_type": mime,
            "captured_at_utc": utc_now(),
            "outcome_fields_read": "",
        }
        row["status"] = recovery_status({**row, "fetch_succeeded": True})
        recovered_rows.append(row)
        audit_rows.append({
            "canonical_event_id": row["canonical_event_id"],
            "url": row["url"],
            "candidate_source_uid": row["candidate_source_uid"],
            "source_type": row["source_type"],
            "fetch_succeeded": True,
            "status": row["status"],
            "outcome_fields_read": "",
        })

    new_recovered = pd.DataFrame(recovered_rows)
    if RECOVERED.is_file():
        old = pd.read_parquet(RECOVERED)
        recovered = pd.concat([old, new_recovered], ignore_index=True, sort=False)
    else:
        recovered = new_recovered
    if not recovered.empty:
        recovered = recovered.sort_values(["canonical_event_id", "url", "captured_at_utc"], kind="mergesort")
        recovered = recovered.drop_duplicates(["canonical_event_id", "url"], keep="last").reset_index(drop=True)
    atomic_parquet(RECOVERED, recovered)

    audit = pd.DataFrame(audit_rows)
    atomic_parquet(FETCH_AUDIT, audit)
    strict = recovered.loc[recovered["strict_independent_media_root_accepted"].fillna(False).astype(bool)] if not recovered.empty else recovered
    official = recovered.loc[recovered["official_root_accepted"].fillna(False).astype(bool)] if not recovered.empty else recovered
    statuses = {str(key): int(value) for key, value in audit["status"].value_counts().items()} if not audit.empty else {}
    report = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_BODY_RECOVERY_REPORT_V1",
        "generated_at_utc": utc_now(),
        "outcome_blind": True,
        "outcome_fields_read": [],
        "candidate_urls": int(len(candidates)),
        "network_requests_including_robots": int(fetcher.requests_used),
        "cross_event_raw_reuses_this_run": int(cross_event_raw_reuses),
        "fetch_successes_this_run": int(sum(bool(row["fetch_succeeded"]) for row in audit_rows)),
        "recovered_articles_total": int(len(recovered)),
        "strict_independent_media_root_events": int(strict["canonical_event_id"].nunique()) if not strict.empty else 0,
        "official_root_events_t2": int(official["canonical_event_id"].nunique()) if not official.empty else 0,
        "status_counts_this_run": statuses,
        "source_freeze_ready": False,
        "model_training_allowed": False,
        "base_v1_mutated": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "artifacts": {
            RECOVERED.name: sha256_file(RECOVERED),
            FETCH_AUDIT.name: sha256_file(FETCH_AUDIT),
        },
    }
    atomic_json(REPORT, report)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update({
        "updated_at_utc": utc_now(),
        "status": "PUBLIC_BODY_RECOVERY_ACTIVE" if recovered_rows else "PUBLIC_BODY_RECOVERY_SOURCE_EXHAUSTED",
        "phase": "OUTCOME_BLIND_ARTICLE_NORMALIZATION_AND_SOURCE_GRAPH",
        "body_recovery_candidate_urls": int(len(candidates)),
        "body_recovery_network_requests": int(fetcher.requests_used),
        "recovered_public_articles": int(len(recovered)),
        "strict_independent_roots_added": report["strict_independent_media_root_events"],
        "official_root_events_t2": report["official_root_events_t2"],
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
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-requests", type=int, default=40)
    args = parser.parse_args()
    if args.max_requests < 1:
        raise SystemExit("--max-requests must be positive")
    print(json.dumps(run(args.max_requests), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
