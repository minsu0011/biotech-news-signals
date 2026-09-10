from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlsplit
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests

from historical_news_multisource_v2_local_recovery import (
    AGGREGATORS,
    lexical_similarity,
    publisher_class,
)


ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "research" / "hist_news_multisource_v2" / "EXACT_EVENT_MULTISOURCE_PRIORITY_QUEUE.parquet"
POLICY_MANIFEST = ROOT / "research" / "multisource_v2" / "POLICY_FREEZE_MANIFEST.json"
RESEARCH = ROOT / "research" / "multisource_v2"
WORKING = ROOT / "data" / "HIST_NEWS_MULTISOURCE_V2_WORKING"
CURRENT_PROVENANCE = WORKING / "V2_CURRENT_SOURCE_PROVENANCE.parquet"
DISCOVERED = WORKING / "V2_DISCOVERED_RSS_ITEMS.parquet"
QUERY_QUEUE = WORKING / "V2_EVENT_QUERY_QUEUE.parquet"
KIND_RECOVERED = WORKING / "V2_RECOVERED_KIND_DISCLOSURES.parquet"
LEDGER = RESEARCH / "COLLECTION_LEDGER.jsonl"
PROVIDER_LEDGER = RESEARCH / "PROVIDER_LEDGER.csv"
REPORT = RESEARCH / "DISCOVERY_REPORT.json"
STATE = ROOT / "state" / "HIST_NEWS_MULTISOURCE_V2_STATE.json"
RAW = WORKING / "raw" / "rss"

USER_AGENT = "MARKET_BIO_HIST_NEWS_RESEARCH/2.0 (public RSS; contact: local-research)"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MIN_HOST_INTERVAL_SECONDS = 1.0

DIRECT_DOMAIN_PUBLISHERS = {
    "etnews.com": "전자신문",
    "edaily.co.kr": "이데일리",
    "fnnews.com": "파이낸셜뉴스",
    "mt.co.kr": "머니투데이",
    "newstap.co.kr": "뉴스탭",
    "newspim.com": "뉴스핌",
    "segye.com": "세계일보",
    "docdocdoc.co.kr": "청년의사",
    "etoday.co.kr": "이투데이",
    "sportalkorea.com": "스포탈코리아",
    "kyeonggi.com": "경기일보",
    "topdaily.kr": "톱데일리",
    "hankyung.com": "한국경제",
    "joongang.co.kr": "중앙일보",
    "inews24.com": "아이뉴스24",
}
AGGREGATOR_DOMAINS = {
    "bing.com", "google.com", "news.google.com", "msn.com", "naver.com",
    "news.naver.com", "n.news.naver.com", "v.daum.net", "yahoo.com",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def query_text(
    headline: Any, ticker: Any, market: Any, mode: str = "HEADLINE_TICKER_V1", event_time: Any = None
) -> str:
    text = re.sub(r"\s+-\s+[^-]{1,40}$", "", str(headline or "")).strip()
    text = text.replace("...", " ").replace("…", " ")
    words = text.split()
    limit = 10 if str(market).upper() == "US" else 8
    phrase = " ".join(words[:limit]).strip()
    if not phrase:
        phrase = str(ticker or "").strip()
    if mode == "EXACT_HEADLINE_ONLY_V2":
        return f'"{phrase}"'.strip()
    if mode == "KIND_OFFICIAL_SPECIFIC_V5":
        specific = text.split("(", 1)[1] if "(" in text else text
        specific = specific.replace(")", " ").replace("ㆍ", " ")
        phrase = " ".join(specific.split()[:12]).strip() or phrase
    if mode in {"ENTITY_TERMS_DATE_V3", "KIND_OFFICIAL_TERMS_DATE_V4", "KIND_OFFICIAL_SPECIFIC_V5"}:
        date = ""
        stamp = pd.to_datetime(event_time, utc=True, errors="coerce")
        if pd.notna(stamp):
            timezone_name = "Asia/Seoul" if str(market).upper() == "KR" else "America/New_York"
            date = stamp.tz_convert(timezone_name).date().isoformat()
        return " ".join(value for value in (phrase, str(ticker or "").strip(), date) if value).strip()
    if mode != "HEADLINE_TICKER_V1":
        raise ValueError(f"unsupported query mode: {mode}")
    return f'"{phrase}" {str(ticker or "").strip()}'.strip()


def inferred_origin(publisher_rss: Any, direct_url: Any) -> tuple[str, str]:
    source = str(publisher_rss or "UNKNOWN").strip()
    try:
        host = (urlsplit(str(direct_url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    for domain, publisher in DIRECT_DOMAIN_PUBLISHERS.items():
        if host == domain or host.endswith("." + domain):
            return publisher, "DIRECT_URL_DOMAIN_REGISTRY"
    if source not in AGGREGATORS and source != "UNKNOWN":
        return source, "RSS_SOURCE_METADATA"
    if host and not any(host == domain or host.endswith("." + domain) for domain in AGGREGATOR_DOMAINS):
        return host, "DIRECT_URL_DOMAIN_FALLBACK"
    return "UNKNOWN", "UNRESOLVED_AGGREGATOR"


def provider_url(provider: str, query: str, market: str) -> str:
    encoded = quote_plus(query)
    if provider == "GOOGLE_NEWS_PUBLIC_RSS":
        if market == "KR":
            return f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"
        return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    if provider == "BING_NEWS_PUBLIC_RSS":
        locale = "ko-kr" if market == "KR" else "en-us"
        return f"https://www.bing.com/news/search?q={encoded}&format=rss&setlang={locale}"
    raise ValueError(f"unsupported provider: {provider}")


def direct_url_from_rss(link: str) -> str:
    split = urlsplit(link)
    query = parse_qs(split.query)
    for key in ("url", "u", "target"):
        value = query.get(key)
        if value and value[0].startswith(("http://", "https://")):
            return unquote(value[0])
    return link


def _text(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else " ".join(element.text.split())


def parse_rss(payload: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = _text(item.find("title"))
        link = _text(item.find("link"))
        published_raw = _text(item.find("pubDate"))
        source = _text(item.find("source"))
        if not source and " - " in title:
            source = title.rsplit(" - ", 1)[-1].strip()
        headline = title
        if source and headline.endswith(" - " + source):
            headline = headline[: -(len(source) + 3)].strip()
        try:
            published = parsedate_to_datetime(published_raw)
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            published_iso = published.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError, OverflowError):
            published_iso = ""
        if title and link:
            rows.append(
                {
                    "headline": headline,
                    "rss_title": title,
                    "rss_url": link,
                    "direct_url_candidate": direct_url_from_rss(link),
                    "publisher_rss": source or "UNKNOWN",
                    "published_at_utc": published_iso,
                    "published_at_source_value": published_raw,
                }
            )
    return rows


class PublicRssClient:
    def __init__(self, provider: str, max_requests: int):
        self.provider = provider
        self.max_requests = max_requests
        self.requests_used = 0
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/rss+xml,application/xml,text/xml;q=0.9"})
        self.robots_checked = False
        self.robots_allowed = False
        self.robots_status = "NOT_CHECKED"

    def _wait(self) -> None:
        remaining = MIN_HOST_INTERVAL_SECONDS - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)

    def _get(self, url: str, timeout: tuple[int, int] = (5, 20)) -> requests.Response | None:
        if self.requests_used >= self.max_requests:
            return None
        self._wait()
        self.requests_used += 1
        self.last_request = time.monotonic()
        try:
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
        except requests.RequestException:
            return None
        if len(response.content) > MAX_RESPONSE_BYTES:
            return None
        return response

    def check_robots(self, sample_url: str) -> bool:
        if self.robots_checked:
            return self.robots_allowed
        split = urlsplit(sample_url)
        robots_url = f"{split.scheme}://{split.netloc}/robots.txt"
        response = self._get(robots_url)
        self.robots_checked = True
        if response is None:
            self.robots_status = "ROBOTS_FETCH_FAILED"
            return False
        if response.status_code in {404, 410}:
            self.robots_allowed = True
            self.robots_status = "NO_DECLARED_RESTRICTION"
            return True
        if response.status_code != 200:
            self.robots_status = f"ROBOTS_HTTP_{response.status_code}"
            return False
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        self.robots_allowed = parser.can_fetch(USER_AGENT, sample_url)
        self.robots_status = "ALLOWED" if self.robots_allowed else "ROBOTS_DISALLOW"
        return self.robots_allowed

    def fetch(self, url: str) -> tuple[bytes | None, dict[str, Any]]:
        if not self.check_robots(url):
            return None, {"status": self.robots_status, "http_status": None}
        response = self._get(url)
        if response is None:
            return None, {"status": "REQUEST_FAILED", "http_status": None}
        if response.status_code == 429:
            return None, {"status": "RATE_LIMITED", "http_status": 429}
        if response.status_code in {401, 403, 404, 410, 451}:
            return None, {"status": "TERMINAL_FOR_PROVIDER", "http_status": response.status_code}
        if response.status_code != 200:
            return None, {"status": "HTTP_FAILED", "http_status": response.status_code}
        mime = response.headers.get("Content-Type", "").lower()
        if not any(value in mime for value in ("xml", "rss")):
            return None, {"status": "INVALID_MIME", "http_status": 200}
        return response.content, {"status": "SUCCESS", "http_status": 200}


def choose_events(queue: pd.DataFrame, current: pd.DataFrame, attempted: set[str], per_market: int) -> pd.DataFrame:
    frame = queue.merge(current, on="canonical_event_id", how="inner", validate="one_to_one")
    frame["event_time_utc"] = pd.to_datetime(frame["event_time_utc"], utc=True, errors="raise")
    frame = frame.loc[
        ~frame["canonical_event_id"].astype(str).isin(attempted)
        & frame["origin_publisher_resolved"].astype(str).ne("UNKNOWN")
    ].copy()
    selected = []
    for market, group in frame.groupby("market", sort=True):
        chosen = group.sort_values(["event_time_utc", "canonical_event_id"], ascending=[False, True], kind="mergesort").head(per_market)
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0].copy()


def choose_kind_events(queue: pd.DataFrame, current: pd.DataFrame, attempted: set[str], limit: int) -> pd.DataFrame:
    if not KIND_RECOVERED.is_file():
        return queue.iloc[0:0].copy()
    official = pd.read_parquet(KIND_RECOVERED)
    official = official.loc[official["official_root_accepted"].fillna(False).astype(bool)].copy()
    if official.empty:
        return queue.iloc[0:0].copy()
    official["published_at_utc"] = pd.to_datetime(official["published_at_utc"], utc=True, errors="coerce")
    official = official.dropna(subset=["published_at_utc"]).sort_values(
        ["canonical_event_id", "published_at_utc", "candidate_source_uid"], kind="mergesort"
    ).drop_duplicates("canonical_event_id", keep="first")
    official = official[["canonical_event_id", "published_at_utc", "headline"]].rename(columns={
        "published_at_utc": "official_event_time_utc", "headline": "query_headline",
    })
    frame = queue.merge(current, on="canonical_event_id", how="inner", validate="one_to_one")
    frame = frame.merge(official, on="canonical_event_id", how="inner", validate="one_to_one")
    frame = frame.loc[~frame["canonical_event_id"].astype(str).isin(attempted)].copy()
    frame["event_time_utc"] = frame["official_event_time_utc"]
    return frame.sort_values(["event_time_utc", "canonical_event_id"], ascending=[False, True], kind="mergesort").head(limit)


def collect_provider(provider: str, events: pd.DataFrame, max_requests: int, query_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    client = PublicRssClient(provider, max_requests=max_requests)
    discoveries: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    successes = 0
    parsed_items = 0
    for event in events.itertuples(index=False):
        query_headline = str(getattr(event, "query_headline", "") or event.current_headline)
        query = query_text(query_headline, event.ticker, event.market, query_mode, event.event_time_utc)
        url = provider_url(provider, query, str(event.market))
        digest = sha256_bytes(canonical_bytes({"provider": provider, "event": str(event.canonical_event_id), "query": query}))
        payload, status = client.fetch(url)
        raw_sha = ""
        item_count = 0
        accepted_count = 0
        if payload is not None:
            raw_sha = sha256_bytes(payload)
            raw_path = RAW / provider.lower() / f"{digest}.xml.gz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if not raw_path.is_file():
                with gzip.open(raw_path, "wb", compresslevel=9) as handle:
                    handle.write(payload)
            try:
                items = parse_rss(payload)
            except ET.ParseError:
                status = {"status": "XML_PARSE_FAILED", "http_status": status.get("http_status")}
                items = []
            item_count = len(items)
            parsed_items += item_count
            event_time = pd.Timestamp(event.event_time_utc)
            cutoff = event_time + pd.Timedelta(minutes=2)
            for item in items:
                published = pd.to_datetime(item["published_at_utc"], utc=True, errors="coerce")
                sequence_current, jaccard_current = lexical_similarity(item["headline"], event.current_headline)
                sequence_query, jaccard_query = lexical_similarity(item["headline"], query_headline)
                sequence = max(sequence_current, sequence_query)
                jaccard = max(jaccard_current, jaccard_query)
                semantic_candidate = bool(sequence >= 0.40 and jaccard >= 0.15)
                in_discovery_window = bool(
                    pd.notna(published)
                    and published >= event_time - pd.Timedelta(hours=24)
                    and published <= event_time + pd.Timedelta(hours=24)
                )
                if not semantic_candidate or not in_discovery_window:
                    continue
                origin, origin_method = inferred_origin(item["publisher_rss"], item["direct_url_candidate"])
                record = {
                    "schema_version": "HIST_NEWS_MULTISOURCE_V2_RSS_DISCOVERY_V1",
                    "canonical_event_id": str(event.canonical_event_id),
                    "market": str(event.market),
                    "ticker": str(event.ticker),
                    "event_time_utc": event_time,
                    "decision_cutoff_utc": cutoff,
                    "current_article_uid": str(event.current_article_uid),
                    "current_origin_publisher": str(event.origin_publisher_resolved),
                    "current_headline": str(event.current_headline),
                    "query_headline": query_headline,
                    "discovery_provider": provider,
                    "discovery_query": query,
                    "discovery_query_mode": query_mode,
                    **item,
                    "published_at_utc": published,
                    "timestamp_precision": "EXACT_SECOND" if pd.notna(published) else "UNKNOWN",
                    "timestamp_provenance": "PUBLIC_RSS_PUBDATE",
                    "timestamp_confidence": "MEDIUM_PENDING_ORIGIN_VERIFICATION",
                    "headline_sequence_ratio": sequence,
                    "headline_char3_jaccard": jaccard,
                    "causal_rss_candidate_t2": bool(pd.notna(published) and published <= cutoff),
                    "origin_distinct_from_current": origin not in AGGREGATORS and origin != str(event.origin_publisher_resolved),
                    "origin_publisher_candidate": origin,
                    "origin_resolution_method": origin_method,
                    "publisher_class": publisher_class(origin),
                    "body_recovery_required": True,
                    "strict_independent_root_accepted": False,
                    "outcome_fields_read": "",
                    "raw_rss_sha256": raw_sha,
                }
                record["discovery_uid"] = "RSS_" + sha256_bytes(
                    canonical_bytes({key: str(record[key]) for key in ("canonical_event_id", "rss_url", "published_at_utc")})
                )[:32]
                discoveries.append(record)
                accepted_count += 1
            if status["status"] == "SUCCESS":
                successes += 1
        ledger.append(
            {
                "attempted_at_utc": utc_now(),
                "provider": provider,
                "canonical_event_id": str(event.canonical_event_id),
                "market": str(event.market),
                "query_digest": digest,
                "query_mode": query_mode,
                "status": status["status"],
                "http_status": status.get("http_status"),
                "rss_items": item_count,
                "accepted_discovery_items": accepted_count,
                "raw_rss_sha256": raw_sha,
                "outcome_fields_read": [],
            }
        )
    provider_summary = {
        "provider": provider,
        "query_mode": query_mode,
        "events_attempted": int(len(events)),
        "requests_used_including_robots": int(client.requests_used),
        "successful_queries": int(successes),
        "parsed_items": int(parsed_items),
        "accepted_discovery_items": int(len(discoveries)),
        "robots_status": client.robots_status,
    }
    return discoveries, ledger, provider_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-per-market", type=int, default=20)
    parser.add_argument("--max-requests-per-provider", type=int, default=50)
    parser.add_argument(
        "--providers", default="GOOGLE_NEWS_PUBLIC_RSS,BING_NEWS_PUBLIC_RSS",
        help="comma-separated public RSS providers",
    )
    parser.add_argument(
        "--query-mode", choices=(
            "HEADLINE_TICKER_V1", "EXACT_HEADLINE_ONLY_V2", "ENTITY_TERMS_DATE_V3",
            "KIND_OFFICIAL_TERMS_DATE_V4", "KIND_OFFICIAL_SPECIFIC_V5",
        ),
        default="HEADLINE_TICKER_V1",
    )
    parser.add_argument("--markets", default="US,KR", help="comma-separated market allowlist")
    args = parser.parse_args()
    if args.events_per_market < 1 or args.max_requests_per_provider < 2:
        raise SystemExit("invalid collection bounds")
    if not POLICY_MANIFEST.is_file() or not CURRENT_PROVENANCE.is_file():
        raise RuntimeError("policy freeze and local provenance recovery are required")
    policy = json.loads(POLICY_MANIFEST.read_text(encoding="utf-8"))
    if not policy.get("outcome_blind") or policy.get("label_join_allowed"):
        raise RuntimeError("collection policy guard failed")

    markets = {value.strip().upper() for value in args.markets.split(",") if value.strip()}
    if not markets or not markets.issubset({"US", "KR"}):
        raise RuntimeError("markets must be a non-empty subset of US,KR")
    queue = pd.read_parquet(QUEUE)
    queue = queue.loc[queue["market"].astype(str).str.upper().isin(markets)].copy()
    current = pd.read_parquet(CURRENT_PROVENANCE)
    prior_ledger = read_jsonl(LEDGER)
    attempted = {
        str(row.get("canonical_event_id")) for row in prior_ledger
        if row.get("canonical_event_id") and row.get("provider") in {"GOOGLE_NEWS_PUBLIC_RSS", "BING_NEWS_PUBLIC_RSS"}
        and str(row.get("query_mode") or "HEADLINE_TICKER_V1") == args.query_mode
    }
    if args.query_mode in {"KIND_OFFICIAL_TERMS_DATE_V4", "KIND_OFFICIAL_SPECIFIC_V5"}:
        if markets != {"KR"}:
            raise RuntimeError("KIND official query modes require --markets KR")
        events = choose_kind_events(queue, current, attempted, args.events_per_market)
    else:
        events = choose_events(queue, current, attempted, args.events_per_market)
    query_frame = events[[
        "canonical_event_id", "market", "ticker", "event_time_utc", "current_article_uid",
        "origin_publisher_resolved", "current_headline",
    ]].copy()
    query_frame["query_headline"] = (
        events["query_headline"].astype(str).to_numpy()
        if "query_headline" in events.columns else query_frame["current_headline"].astype(str)
    )
    query_frame["query_text"] = [
        query_text(row.query_headline, row.ticker, row.market, args.query_mode, row.event_time_utc)
        for row in query_frame.itertuples(index=False)
    ]
    query_frame["query_mode"] = args.query_mode
    query_frame["outcome_fields_read"] = ""
    atomic_parquet(QUERY_QUEUE, query_frame)

    providers = [value.strip() for value in args.providers.split(",") if value.strip()]
    allowed = {"GOOGLE_NEWS_PUBLIC_RSS", "BING_NEWS_PUBLIC_RSS"}
    if not providers or any(value not in allowed for value in providers):
        raise RuntimeError("provider not allowlisted")
    results = []
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = [
            executor.submit(collect_provider, provider, events, args.max_requests_per_provider, args.query_mode)
            for provider in providers
        ]
        for future in futures:
            results.append(future.result())

    new_discoveries = [row for discoveries, _, _ in results for row in discoveries]
    new_ledger = [row for _, ledger, _ in results for row in ledger]
    provider_summaries = [summary for _, _, summary in results]
    append_jsonl(LEDGER, new_ledger)

    new_frame = pd.DataFrame(new_discoveries)
    prior_discovery_ids: set[str] = set()
    if DISCOVERED.is_file():
        old = pd.read_parquet(DISCOVERED)
        prior_discovery_ids = set(old["discovery_uid"].astype(str)) if "discovery_uid" in old else set()
        combined = pd.concat([old, new_frame], ignore_index=True) if not new_frame.empty else old
    else:
        combined = new_frame
    if not combined.empty:
        combined = combined.sort_values(
            ["canonical_event_id", "published_at_utc", "discovery_uid"], kind="mergesort"
        ).drop_duplicates("discovery_uid", keep="first").reset_index(drop=True)
        resolved_origins = [
            inferred_origin(row.publisher_rss, row.direct_url_candidate)
            for row in combined.itertuples(index=False)
        ]
        combined["origin_publisher_candidate"] = [value[0] for value in resolved_origins]
        combined["origin_resolution_method"] = [value[1] for value in resolved_origins]
        combined["origin_distinct_from_current"] = [
            origin != "UNKNOWN" and origin != str(current_origin)
            for origin, current_origin in zip(
                combined["origin_publisher_candidate"], combined["current_origin_publisher"]
            )
        ]
        combined["publisher_class"] = [publisher_class(value) for value in combined["origin_publisher_candidate"]]
    atomic_parquet(DISCOVERED, combined)
    new_unique_count = int(len(set(new_frame["discovery_uid"].astype(str)) - prior_discovery_ids)) if not new_frame.empty else 0

    provider_frame = pd.DataFrame(provider_summaries)
    provider_frame["audited_at_utc"] = utc_now()
    provider_frame["credential_values_serialized"] = False
    provider_frame.to_csv(PROVIDER_LEDGER, index=False, encoding="utf-8", lineterminator="\n")

    distinct_causal = combined.loc[
        combined["causal_rss_candidate_t2"].fillna(False).astype(bool)
        & combined["origin_distinct_from_current"].fillna(False).astype(bool)
    ] if not combined.empty else combined
    report = {
        "schema_version": "HIST_NEWS_MULTISOURCE_V2_DISCOVERY_REPORT_V1",
        "generated_at_utc": utc_now(),
        "outcome_blind": True,
        "outcome_fields_read": [],
        "events_attempted_this_run": int(len(events)),
        "query_mode": args.query_mode,
        "markets_requested": sorted(markets),
        "events_by_market_this_run": {str(key): int(value) for key, value in events["market"].value_counts().items()},
        "provider_summaries": provider_summaries,
        "new_discovery_items": int(len(new_frame)),
        "new_discovery_items_unique": new_unique_count,
        "total_discovery_items": int(len(combined)),
        "causal_distinct_origin_candidate_events": int(distinct_causal["canonical_event_id"].nunique()) if not combined.empty else 0,
        "strict_independent_roots_added": 0,
        "body_recovery_required": int(combined["body_recovery_required"].sum()) if not combined.empty else 0,
        "network_requests_this_run": int(sum(row["requests_used_including_robots"] for row in provider_summaries)),
        "collection_status": "DISCOVERY_CONTINUES_BODY_NOT_VERIFIED",
        "source_freeze_ready": False,
        "model_training_allowed": False,
        "base_v1_mutated": False,
        "research_seal_touched": False,
        "final_meta_touched": False,
        "v224_touched": False,
        "artifacts": {
            DISCOVERED.name: sha256_file(DISCOVERED),
            QUERY_QUEUE.name: sha256_file(QUERY_QUEUE),
            PROVIDER_LEDGER.name: sha256_file(PROVIDER_LEDGER),
        },
    }
    atomic_json(REPORT, report)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    previous_requests = int(state.get("collection_network_requests", 0))
    previous_zero = int(state.get("consecutive_zero_discovery_cycles", 0))
    state.update(
        {
            "updated_at_utc": utc_now(),
            "status": "PUBLIC_RSS_DISCOVERY_ACTIVE" if new_unique_count else "PUBLIC_RSS_NO_PROGRESS_SWITCH_SOURCE_FAMILY",
            "phase": "OUTCOME_BLIND_ALTERNATE_SOURCE_COLLECTION",
            "events_attempted": int(state.get("events_attempted", 0)) + int(len(events)),
            "collection_network_requests": previous_requests + report["network_requests_this_run"],
            "rss_discovery_items": int(len(combined)),
            "causal_distinct_origin_candidate_events": report["causal_distinct_origin_candidate_events"],
            "strict_independent_roots_added": int(state.get("strict_independent_roots_added", 0)),
            "consecutive_zero_discovery_cycles": 0 if new_unique_count else previous_zero + 1,
            "labels_read": False,
            "source_frozen": False,
            "split_frozen": False,
            "model_training_allowed": False,
            "base_v1_membership_mutable": False,
            "research_seal_touched": False,
            "final_meta_touched": False,
            "v224_touched": False,
        }
    )
    atomic_json(STATE, state)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
