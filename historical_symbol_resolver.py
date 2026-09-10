"""CIK-centred historical ticker recovery and Alpaca SIP backfill.

Resolution is identity-only metadata work.  It never reads an outcome column,
never changes a frozen role, and records whether evidence was available at the
event date or is post-hoc.  Only Tier A/B evidence is eligible for the exact
price pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from lxml import html as lxml_html

import alpaca_backfill_v221 as alpaca_backfill
import alpaca_provider_acquisition as alpaca
import bio_news_30m_v3 as app
import v59_provider_acquisition as provider
from provider_env_runtime import ALPACA_ENV_NAMES, refresh_user_environment


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
CACHE = ROOT / "cache" / "historical_symbol_resolver"
QUEUE_PATH = DATA / "HISTORICAL_SYMBOL_BACKFILL_QUEUE.parquet"
ALIAS_PATH = DATA / "HISTORICAL_TICKER_ALIAS_REGISTRY.parquet"
TIMELINE_PATH = DATA / "CIK_SYMBOL_TIMELINE.parquet"
REPORT_PATH = RESEARCH / "HISTORICAL_SYMBOL_RECOVERY_REPORT.json"
LEDGER_PATH = RESEARCH / "HISTORICAL_RECOVERY_LEDGER.csv"
CONSENSUS_POLICY_PATH = DATA / "HISTORICAL_SYMBOL_CONSENSUS_POLICY_V1.json"
CONSENSUS_AUDIT_PATH = RESEARCH / "HISTORICAL_SYMBOL_CONSENSUS_AUDIT.json"
ALPACA_NORMAL_RETRY_POLICY_PATH = DATA / "ALPACA_RESOLVED_SYMBOL_RETRY_POLICY_V1.json"
ALPACA_NORMAL_RETRY_REPORT_PATH = RESEARCH / "ALPACA_RESOLVED_SYMBOL_NORMAL_RETRY_REPORT.json"
NO_BAR_CLASSIFICATION_PATH = DATA / "HISTORICAL_NO_BAR_CLASSIFICATION.parquet"
US_QUEUE_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US.parquet"
ROLE_PATH = DATA / "V59_DATA_ROLE_ASSIGNMENT.parquet"
PROVIDER_STATUS_PATH = DATA / "PRICE_ACQUISITION_QUEUE_US_PROVIDER_STATUS.parquet"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
ARCHIVE_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
TARGET_YEARS = set(range(2018, 2025))
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
EXACT_CONFIDENCE = {"A", "B"}
COMMON_TITLE = re.compile(
    r"(?i)\b(?:common\s+(?:stock|shares?)|ordinary\s+shares?|"
    r"american\s+depositary\s+(?:shares?|receipts?)|ADSs?|ADRs?)\b"
)
EXCLUDED_TITLE = re.compile(
    r"(?i)\b(?:warrants?|preferred|units?|rights?|notes?|debentures?)\b"
)
SYMBOL_TOKEN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
TRADING_SYMBOL_HEADER = re.compile(
    r"(?i)(?:trading|ticker)\s+symbol(?:\(s\))?"
    r"(?:\s*\(\d+\)|\s*[*\u2020\u2021]+)?\s*:?[\s.]*"
)
INVALID_SYMBOLS = {
    "A", "AN", "AND", "AS", "AT", "BY", "COMMON", "EACH", "EXCHANGE",
    "NAME", "NASDAQ", "NONE", "NYSE", "OF", "ON", "OR", "STOCK", "SYMBOL",
    "THE", "TITLE", "TRADING", "N/A", "NA",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(temporary: Path, path: Path) -> None:
    """Commit an atomic file despite brief Windows reader locks."""
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.2)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    replace_with_retry(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    replace_with_retry(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    replace_with_retry(temporary, path)


def parse_event_id(event_id: str) -> tuple[str, str]:
    parts = str(event_id).split(":", 2)
    if len(parts) != 3 or parts[0] != "SEC" or not parts[1].isdigit():
        return "", ""
    return str(int(parts[1])), parts[2]


def normalize_symbol(value: Any) -> str:
    symbol = re.sub(r"\s+", "", str(value or "").upper()).replace(".", "-")
    symbol = symbol.strip("'\"()[]{}:,;*-_")
    if not SYMBOL_TOKEN.fullmatch(symbol) or symbol in INVALID_SYMBOLS:
        return ""
    return symbol


def title_score(title: str) -> int:
    if EXCLUDED_TITLE.search(title or ""):
        return -20
    if COMMON_TITLE.search(title or ""):
        return 40
    return 5


def symbol_tokens(value: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9.-]{0,9}", value or ""):
        symbol = normalize_symbol(raw)
        if symbol and symbol not in tokens:
            tokens.append(symbol)
    return tokens


def extract_trading_symbols(raw: str) -> list[dict[str, Any]]:
    """Extract DEI and cover-page symbol evidence from one filing."""
    if not raw:
        return []
    # Use lxml directly. BeautifulSoup's wrapper is more than an order of
    # magnitude slower on multi-megabyte 10-K/20-F documents.
    document = lxml_html.fromstring(raw.encode("utf-8", errors="replace"))
    title_by_context: dict[str, str] = {}
    facts: list[dict[str, Any]] = []
    no_symbol = False
    # iXBRL facts carry a name attribute. Namespace XML facts without one are
    # covered by the raw regex fallback below, so do not iterate every HTML
    # element in a large filing twice.
    tags = document.xpath("//*[@name]")
    for tag in tags:
        tag_name = str(tag.tag or "").lower()
        fact_name = str(tag.attrib.get("name", tag_name)).lower().replace("-", "")
        if not fact_name.endswith(("notradingsymbolflag", "security12btitle", "security12gtitle")):
            continue
        context = str(tag.attrib.get("contextref", ""))
        text = " ".join(part.strip() for part in tag.itertext() if part.strip())
        if fact_name.endswith("notradingsymbolflag") and text.lower() in {"true", "1"}:
            no_symbol = True
        if fact_name.endswith("security12btitle") or fact_name.endswith("security12gtitle"):
            title_by_context[context] = text
    for tag in tags:
        tag_name = str(tag.tag or "").lower()
        fact_name = str(tag.attrib.get("name", tag_name)).lower().replace("-", "")
        if not fact_name.endswith("tradingsymbol"):
            continue
        context = str(tag.attrib.get("contextref", ""))
        text = " ".join(part.strip() for part in tag.itertext() if part.strip())
        if fact_name.endswith("tradingsymbol"):
            for symbol in symbol_tokens(text):
                title = title_by_context.get(context, "")
                facts.append({
                    "symbol": symbol,
                    "method": "IXBRL_DEI_TRADING_SYMBOL",
                    "context": context,
                    "security_title": title,
                    "score": 100 + title_score(title),
                })
    # Some XML parsers expose namespace tags poorly; retain a raw fallback.
    patterns = (
        r"(?is)<[^>]*\bname=[\"']dei:TradingSymbol[\"'][^>]*>(.*?)</[^>]+>",
        r"(?is)<dei:TradingSymbol[^>]*>(.*?)</dei:TradingSymbol>",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, raw):
            text = BeautifulSoup(match.group(1), "html.parser").get_text(" ", strip=True)
            for symbol in symbol_tokens(text):
                facts.append({"symbol": symbol, "method": "XBRL_DEI_TRADING_SYMBOL",
                              "context": "", "security_title": "", "score": 100})
    # Native ownership XML (Forms 3/4/5) carries an issuer trading symbol
    # even when the rendered bracketed cover line is absent from the primary
    # document. This is issuer identity metadata, not a model feature.
    ownership_patterns = (
        r"(?is)<issuerTradingSymbol[^>]*>(.*?)</issuerTradingSymbol>",
        r"(?is)<ISSUER-TRADING-SYMBOL[^>]*>(.*?)</ISSUER-TRADING-SYMBOL>",
    )
    for pattern in ownership_patterns:
        for match in re.finditer(pattern, raw):
            value = BeautifulSoup(match.group(1), "html.parser").get_text(" ", strip=True)
            for symbol in symbol_tokens(value):
                facts.append({
                    "symbol": symbol,
                    "method": "SEC_OWNERSHIP_ISSUER_TRADING_SYMBOL",
                    "context": "",
                    "security_title": "Issuer common equity",
                    "score": 115,
                })
    # Cover-page tables: locate the Trading Symbol(s) header and read the same
    # column from following security rows.
    # Locate the few text nodes that actually say Trading Symbol, then walk to
    # their table. Calling get_text() on every nested financial table becomes
    # quadratic on large annual reports.
    candidate_tables = []
    seen_table_ids = set()
    cover_document = document
    if len(raw) > 2_000_000:
        cover_document = lxml_html.fromstring(
            raw[:2_000_000].encode("utf-8", errors="replace")
        )
    cover_text_nodes = cover_document.xpath("//text()")[:20_000]
    for node in cover_text_nodes:
        if not re.search(r"(?i)\b(?:trading|symbol(?:\(s\))?)\b", str(node)):
            continue
        parent = node.getparent()
        table = next((ancestor for ancestor in [parent, *parent.iterancestors()]
                      if str(ancestor.tag).lower().endswith("table")), None)
        if table is not None and id(table) not in seen_table_ids:
            seen_table_ids.add(id(table))
            candidate_tables.append(table)
        if len(candidate_tables) >= 30:
            break
    for table in candidate_tables:
        rows = table.xpath(".//tr")
        for position, table_row in enumerate(rows):
            cells = table_row.xpath("./th|./td")
            texts = [" ".join(part.strip() for part in cell.itertext() if part.strip())
                     for cell in cells]
            header_index = next((i for i, text in enumerate(texts)
                                 if TRADING_SYMBOL_HEADER.fullmatch(text.strip())), None)
            if header_index == 0 and len(texts) >= 2:
                # Older filings also use a vertical key/value cover layout:
                # ``Trading Symbol(s) | ABCD``. A symbol-like entire second
                # cell is required, so a conventional header row stays safe.
                vertical_symbol = normalize_symbol(texts[1])
                if vertical_symbol:
                    facts.append({
                        "symbol": vertical_symbol,
                        "method": "SEC_COVER_TRADING_SYMBOL_KEY_VALUE",
                        "context": "",
                        "security_title": "",
                        "score": 85,
                    })
                    continue
            if header_index is None:
                continue
            for following in rows[position + 1:position + 7]:
                values = [" ".join(part.strip() for part in cell.itertext() if part.strip())
                          for cell in following.xpath("./th|./td")]
                if header_index >= len(values):
                    continue
                title = values[0] if values else ""
                symbol = normalize_symbol(values[header_index])
                for symbol in ([symbol] if symbol else []):
                    facts.append({
                        "symbol": symbol,
                        "method": "SEC_COVER_TRADING_SYMBOL_TABLE",
                        "context": "",
                        "security_title": title,
                        "score": 80 + title_score(title),
                    })
    text = re.sub(r"\s+", " ", " ".join(document.itertext()))
    # Older foreign-issuer forms often describe the common ADS and a listed
    # warrant in one sentence instead of using a modern cover-page symbol
    # table. Preserve the order explicitly so a warrant ticker cannot win
    # merely because it is the only generic "under the symbol" match.
    paired_patterns = (
        r"(?s)(?i:(American\s+Depositary\s+Shares?|ADSs?|common\s+stock|ordinary\s+shares?))"
        r"\s*(?:,?\s+and\s+)(?:(?i:our|the|public|class\s+[ab])\s+)*"
        r"(?i:(warrants?|units?|rights?)).{0,260}?(?i:under\s+the\s+symbols?)"
        r"[^A-Za-z0-9]{0,20}([A-Z][A-Z0-9.-]{0,9})"
        r"[^A-Za-z0-9]{1,20}(?:(?i:and)[^A-Za-z0-9]{1,20})?"
        r"([A-Z][A-Z0-9.-]{0,9}(?:\s+WS)?)",
    )
    for pattern in paired_patterns:
        for match in re.finditer(pattern, text):
            between_titles = text[match.end(1):match.start(2)]
            if re.search(r"(?i)under\s+the\s+(?:trading\s+)?symbol", between_titles):
                continue
            tail = text[match.end():match.end() + 80]
            tail = re.split(r"(?i)respectively", tail, maxsplit=1)[0]
            if symbol_tokens(tail):
                # This was a three-or-more security list. Pair alignment is
                # unsafe because filings do not consistently order those
                # lists; a nearby explicit two-security sentence is required.
                continue
            common_symbol = normalize_symbol(match.group(3))
            excluded_symbol = normalize_symbol(match.group(4))
            cover_bonus = 5 if match.start() < 10_000 else 0
            if common_symbol:
                facts.append({
                    "symbol": common_symbol,
                    "method": "SEC_PAIRED_SECURITY_SYMBOL_TEXT",
                    "context": "",
                    "security_title": match.group(1),
                    "score": 120 + cover_bonus,
                })
            if excluded_symbol:
                facts.append({
                    "symbol": excluded_symbol,
                    "method": "SEC_PAIRED_SECURITY_SYMBOL_TEXT",
                    "context": "",
                    "security_title": match.group(2),
                    "score": 60,
                })
    explicit_patterns = (
        (r"(?is:(?:Nasdaq|NYSE|OTC(?:QB|QX|BB|\s+pink\s+sheets?))"
         r"[^.]{0,160}?under\s+the\s+(?:trading\s+|ticker\s+)?symbol)"
         r"\s*[^A-Za-z0-9]{0,12}([A-Z][A-Z0-9.-]{0,9})", 82),
        (r"(?is:(?:trading|ticker)\s+symbol(?:\(s\))?)\s+"
         r"([A-Z][A-Z0-9.-]{0,9})\s+(?:Nasdaq|NYSE|NYSE\s+American|OTCQB|OTCQX)", 78),
        (r"(?is:changed\s+(?:our\s+)?trading\s+symbol\s+from).{0,80}?"
         r"(?:to|and\s+now\s+trades\s+under)\s*[^A-Za-z0-9]{0,12}"
         r"([A-Z][A-Z0-9.-]{0,9})", 90),
        (r"(?is:Issuer\s+Name\s+and\s+Ticker\s+or\s+Trading\s+Symbol).{0,300}?\[\s*([A-Z][A-Z0-9.-]{0,9})\s*\]", 95),
        (r"(?is:changed\s+(?:our\s+)?trading\s+symbol\s+from).{0,80}?(?:to|and\s+now\s+trades\s+under)\s*[\"'“”]?([A-Z][A-Z0-9.-]{0,9})", 90),
        (r"(?is:(?:Nasdaq|NYSE)[^.]{0,100}?under\s+the\s+(?:trading\s+)?symbol)\s*[\"'“”]?([A-Z][A-Z0-9.-]{0,9})", 80),
        (r"(?is:listed\s+on\s+(?:the\s+)?(?:Nasdaq|NYSE)[^.]{0,100}?symbol)\s*[\"'“”]?([A-Z][A-Z0-9.-]{0,9})", 80),
        (r"(?is:(?:trading|ticker)\s+symbol(?:\(s\))?\s*[:\-])\s*[\"'“”]?([A-Z][A-Z0-9.-]{0,9})(?![a-z])", 70),
    )
    for pattern, score in explicit_patterns:
        for match in re.finditer(pattern, text):
            # Generic prose about an exchange frequently names warrants,
            # units, or rights. It is identity evidence, but not evidence for
            # the common equity whose event return is being labelled.
            nearby = text[max(0, match.start() - 100):match.end()]
            common_hits = list(COMMON_TITLE.finditer(nearby))
            excluded_hits = list(EXCLUDED_TITLE.finditer(nearby))
            if excluded_hits and (not common_hits or excluded_hits[-1].start() > common_hits[-1].start()):
                continue
            symbol = normalize_symbol(match.group(1))
            if symbol:
                facts.append({"symbol": symbol, "method": "SEC_COVER_TEXT_EXPLICIT",
                              "context": "", "security_title": "", "score": score})
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fact in facts:
        key = (fact["symbol"], fact["method"], fact.get("security_title", ""))
        if key not in unique or fact["score"] > unique[key]["score"]:
            unique[key] = fact
    output = sorted(unique.values(), key=lambda row: (-int(row["score"]), row["symbol"], row["method"]))
    if no_symbol and not output:
        output.append({"symbol": "", "method": "IXBRL_NO_TRADING_SYMBOL_FLAG",
                       "context": "", "security_title": "", "score": 0})
    return output


def choose_symbol(evidence: list[dict[str, Any]], queue_ticker: str) -> tuple[str, bool]:
    queue = normalize_symbol(queue_ticker)
    warrant_target = bool(re.search(r"(?:WS|W)$", queue))
    if warrant_target:
        usable = [
            row for row in evidence
            if row.get("symbol") and (
                re.search(r"(?i)warrant", str(row.get("security_title", "")))
                or re.search(r"(?:WS|W)$", normalize_symbol(row.get("symbol")))
            )
        ]
    else:
        usable = [
            row for row in evidence
            if row.get("symbol") and not EXCLUDED_TITLE.search(str(row.get("security_title", "")))
        ]
    if not usable:
        return "", False
    by_symbol: dict[str, int] = {}
    for row in usable:
        symbol = normalize_symbol(row.get("symbol"))
        if symbol:
            by_symbol[symbol] = max(by_symbol.get(symbol, -999), int(row.get("score", 0)))
    if not by_symbol:
        return "", False
    ranked = sorted(by_symbol.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return ranked[0][0], False
    top = {symbol for symbol, score in ranked if score == ranked[0][1]}
    return (queue, False) if queue in top else ("", True)


class SECArchiveClient:
    def __init__(self, min_interval: float = 0.16):
        self.session = requests.Session()
        self.session.headers.update(app.SEC_HEADERS)
        self.min_interval = max(0.11, float(min_interval))
        self.last_request = 0.0
        self.network_requests = 0
        self.cache_hits = 0
        self.evidence_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], Path, str]] = {}

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, cache_path: Path) -> bytes:
        if cache_path.is_file():
            self.cache_hits += 1
            return cache_path.read_bytes()
        self._throttle()
        response = self.session.get(url, timeout=45)
        self.last_request = time.monotonic()
        self.network_requests += 1
        response.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
        temporary.write_bytes(response.content)
        os.replace(temporary, cache_path)
        return response.content

    def json(self, url: str, cache_path: Path) -> dict[str, Any]:
        return json.loads(self.get(url, cache_path).decode("utf-8"))

    def submissions(self, cik: str) -> dict[str, Any]:
        return self.json(
            SUBMISSIONS.format(cik=int(cik)),
            CACHE / "submissions" / f"CIK{int(cik):010d}.json",
        )

    @staticmethod
    def _columnar_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not payload:
            return []
        length = len(payload.get("accessionNumber", []))
        keys = list(payload)
        return [
            {key: (payload.get(key, [])[index] if index < len(payload.get(key, [])) else None)
             for key in keys}
            for index in range(length)
        ]

    def filing_rows(self, cik: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        submission = self.submissions(cik)
        filings = submission.get("filings", {}) or {}
        rows = self._columnar_rows(filings.get("recent", {}) or {})
        for file_row in filings.get("files", []) or []:
            name = str(file_row.get("name", ""))
            if not name:
                continue
            payload = self.json(SUBMISSIONS_FILE.format(name=quote(name)),
                                CACHE / "submissions" / name)
            rows.extend(self._columnar_rows(payload))
        deduplicated = {str(row.get("accessionNumber")): row for row in rows
                        if row.get("accessionNumber")}
        return submission, list(deduplicated.values())

    def archive_index(self, cik: str, accession: str) -> dict[str, Any]:
        compact = re.sub(r"[^0-9]", "", accession)
        return self.json(
            ARCHIVE_INDEX.format(cik=int(cik), accession=compact),
            CACHE / "index" / f"{int(cik)}_{compact}.json",
        )

    def primary_document(self, cik: str, accession: str, filing: dict[str, Any]) -> str:
        primary = str(filing.get("primaryDocument") or "").strip()
        if primary:
            return primary
        try:
            items = (self.archive_index(cik, accession).get("directory", {}) or {}).get("item", []) or []
        except Exception:
            return ""
        candidates = []
        for item in items:
            name = str(item.get("name", ""))
            lower = name.lower()
            if not re.search(r"\.(?:htm|html|txt)$", lower):
                continue
            score = int(item.get("size") or 0)
            if "index" in lower or "filingsummary" in lower or lower.startswith("r"):
                score -= 10_000_000
            candidates.append((score, name))
        return max(candidates, default=(0, ""))[1]

    def document(self, cik: str, accession: str, filing: dict[str, Any]) -> tuple[str, Path, str]:
        document = self.primary_document(cik, accession, filing)
        if not document:
            return "", Path(), ""
        compact = re.sub(r"[^0-9]", "", accession)
        safe_document = re.sub(r"[^A-Za-z0-9_.-]", "_", document)
        path = CACHE / "documents" / f"{int(cik)}_{compact}_{safe_document}"
        raw = self.get(
            ARCHIVE.format(cik=int(cik), accession=compact, document=quote(document)),
            path,
        )
        return raw.decode("utf-8", errors="replace"), path, document


def filing_time(row: dict[str, Any]) -> pd.Timestamp | None:
    for key in ("acceptanceDateTime", "filingDate"):
        value = row.get(key)
        if not value:
            continue
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                if key == "acceptanceDateTime" and "T" in str(value):
                    stamp = stamp.tz_localize("America/New_York")
                else:
                    stamp = stamp.tz_localize("UTC")
            return stamp.tz_convert("UTC")
        except Exception:
            continue
    return None


def inspect_filing_evidence(
    cik: str,
    accession: str,
    filing: dict[str, Any],
    client: SECArchiveClient,
) -> tuple[list[dict[str, Any]], Path, str]:
    cache_key = (cik, accession)
    if cache_key in client.evidence_cache:
        return client.evidence_cache[cache_key]
    raw, path, _ = client.document(cik, accession, filing)
    result = (extract_trading_symbols(raw), path, sha256(path) if path.is_file() else "")
    client.evidence_cache[cache_key] = result
    return result


def load_consensus_policy() -> tuple[dict[str, Any], str]:
    if not CONSENSUS_POLICY_PATH.is_file():
        raise RuntimeError("HISTORICAL_SYMBOL_CONSENSUS_POLICY_MISSING")
    policy = json.loads(CONSENSUS_POLICY_PATH.read_text(encoding="utf-8"))
    required_flags = {
        "frozen_before_consensus_results": True,
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "same_cik_required": True,
        "future_filing_evidence_forbidden": True,
        "exact_price_contract_unchanged": True,
    }
    if any(policy.get(key) is not expected for key, expected in required_flags.items()):
        raise RuntimeError("HISTORICAL_SYMBOL_CONSENSUS_POLICY_GUARD_FAIL")
    return policy, sha256(CONSENSUS_POLICY_PATH)


def evaluate_consensus_records(
    records: list[dict[str, Any]],
    ambiguous_filings: int,
    policy: dict[str, Any],
) -> tuple[str, str]:
    """Apply the frozen label-blind CIK consensus contract."""
    if ambiguous_filings and policy["ambiguous_prior_symbol_evidence_forbidden"]:
        return "", "AMBIGUOUS_PRIOR_SYMBOL_EVIDENCE"
    unique_by_accession = {
        str(record["accession"]): record for record in records if record.get("accession")
    }
    records = list(unique_by_accession.values())
    symbols = sorted({normalize_symbol(record.get("symbol")) for record in records}
                     - {""})
    if len(symbols) > 1 and policy["contradictory_prior_symbol_evidence_forbidden"]:
        return "", "CONTRADICTORY_PRIOR_SYMBOL_EVIDENCE"
    if len(symbols) != 1:
        return "", "NO_SINGLE_CONSENSUS_SYMBOL"
    matching = [record for record in records if normalize_symbol(record.get("symbol")) == symbols[0]]
    if len(matching) < int(policy["minimum_independent_prior_filings"]):
        return "", "INSUFFICIENT_INDEPENDENT_PRIOR_FILINGS"
    days = [float(record["days_before_event"]) for record in matching]
    if min(days) > float(policy["maximum_nearest_prior_days"]):
        return "", "NEAREST_PRIOR_OUTSIDE_CONSENSUS_WINDOW"
    if max(days) > float(policy["maximum_oldest_prior_days"]):
        return "", "OLDEST_PRIOR_OUTSIDE_CONSENSUS_WINDOW"
    if policy["evidence_file_sha256_required"] and any(
        not str(record.get("evidence_sha256", "")) for record in matching
    ):
        return "", "CONSENSUS_EVIDENCE_SHA_MISSING"
    return symbols[0], "PASS"


def consensus_columns() -> dict[str, Any]:
    return {
        "corroborating_evidence_files": "[]",
        "corroborating_evidence_sha256": "[]",
        "corroborating_accessions": "[]",
        "consensus_evidence_count": 0,
        "consensus_contract_sha256": "",
        "consensus_status": "NOT_ATTEMPTED",
    }


def ensure_consensus_columns(queue: pd.DataFrame) -> pd.DataFrame:
    for column, default in consensus_columns().items():
        if column not in queue.columns:
            queue[column] = default
    return queue


def corroborate_one(
    row: pd.Series,
    client: SECArchiveClient,
    policy: dict[str, Any],
    policy_sha256: str,
) -> dict[str, Any]:
    cik, event_accession = parse_event_id(str(row.event_id))
    base = {
        "promoted": False,
        "consensus_status": "NOT_ELIGIBLE",
        "consensus_contract_sha256": policy_sha256,
        "corroborating_evidence_files": "[]",
        "corroborating_evidence_sha256": "[]",
        "corroborating_accessions": "[]",
        "consensus_evidence_count": 0,
    }
    if not cik or not event_accession:
        return {**base, "consensus_status": "CIK_OR_ACCESSION_MISSING"}
    if bool(row.get("exact_eligible", False)):
        return {**base, "consensus_status": "ALREADY_EXACT"}
    if str(row.get("form", "")).upper().startswith(("S-1", "F-1")):
        return {**base, "consensus_status": "REGISTRATION_WITHOUT_EVENT_LISTING_FORBIDDEN"}
    try:
        _, filings = client.filing_rows(cik)
    except Exception as error:
        return {**base, "consensus_status": f"SEC_SUBMISSIONS_{type(error).__name__}"}
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
    prior: list[tuple[float, dict[str, Any]]] = []
    for filing in filings:
        accession = str(filing.get("accessionNumber") or "")
        stamp = filing_time(filing)
        if not accession or accession == event_accession or stamp is None or stamp >= event_time:
            continue
        days = (event_time - stamp).total_seconds() / 86400.0
        if 0 <= days <= float(policy["maximum_oldest_prior_days"]):
            prior.append((days, filing))
    prior = sorted(prior, key=lambda item: item[0])[: int(policy["maximum_filings_scanned"])]
    records: list[dict[str, Any]] = []
    ambiguous_filings = 0
    for days, filing in prior:
        accession = str(filing.get("accessionNumber"))
        try:
            evidence, path, digest = inspect_filing_evidence(cik, accession, filing, client)
        except Exception:
            continue
        symbol, ambiguous = choose_symbol(evidence, str(row.ticker))
        if ambiguous:
            ambiguous_filings += 1
            continue
        if not symbol or not path.is_file() or not digest:
            continue
        records.append({
            "accession": accession,
            "filing_time_utc": str(filing_time(filing) or ""),
            "days_before_event": round(days, 6),
            "symbol": symbol,
            "evidence_file": str(path.relative_to(ROOT)),
            "evidence_sha256": digest,
            "evidence_methods": sorted({str(item.get("method", "")) for item in evidence}),
            "evidence": evidence,
        })
    symbol, reason = evaluate_consensus_records(records, ambiguous_filings, policy)
    matching = sorted(
        [record for record in records if normalize_symbol(record.get("symbol")) == symbol],
        key=lambda record: float(record["days_before_event"]),
    ) if symbol else []
    audit_records = matching if matching else records
    evidence_files = [record["evidence_file"] for record in audit_records]
    evidence_hashes = [record["evidence_sha256"] for record in audit_records]
    accessions = [record["accession"] for record in audit_records]
    result = {
        **base,
        "consensus_status": reason,
        "corroborating_evidence_files": json.dumps(evidence_files, ensure_ascii=False),
        "corroborating_evidence_sha256": json.dumps(evidence_hashes, ensure_ascii=False),
        "corroborating_accessions": json.dumps(accessions, ensure_ascii=False),
        "consensus_evidence_count": len(matching),
    }
    if not symbol:
        return result
    nearest = matching[0]
    return {
        **result,
        "promoted": True,
        "resolved_ticker": symbol,
        "ticker_candidates": json.dumps(nearest["evidence"], ensure_ascii=False, sort_keys=True),
        "resolution_method": policy["promoted_resolution_method"],
        "resolution_confidence": policy["promoted_resolution_confidence"],
        "resolution_asof_mode": policy["promoted_resolution_asof_mode"],
        "evidence_file": nearest["evidence_file"],
        "evidence_sha256": nearest["evidence_sha256"],
        "evidence_accession": nearest["accession"],
        "evidence_filing_time_utc": nearest["filing_time_utc"],
        "resolution_error": "",
    }


def resolve_one(row: pd.Series, client: SECArchiveClient) -> dict[str, Any]:
    cik, accession = parse_event_id(str(row.event_id))
    base = {
        "resolved_ticker": "",
        "ticker_candidates": "[]",
        "resolution_method": "UNRESOLVED",
        "resolution_confidence": "D",
        "resolution_asof_mode": "EVENT_DATE_METADATA_ONLY",
        "evidence_file": "",
        "evidence_sha256": "",
        "evidence_accession": "",
        "evidence_filing_time_utc": "",
        "resolution_error": "",
    }
    if not cik or not accession:
        base["resolution_error"] = "CIK_OR_ACCESSION_MISSING"
        return base
    try:
        submission, filings = client.filing_rows(cik)
    except Exception as error:
        base["resolution_error"] = f"SEC_SUBMISSIONS:{type(error).__name__}"
        return base
    by_accession = {str(item.get("accessionNumber")): item for item in filings}

    def inspect(candidate_accession: str, filing: dict[str, Any]) -> tuple[list[dict[str, Any]], Path, str]:
        cache_key = (cik, candidate_accession)
        if cache_key in client.evidence_cache:
            return client.evidence_cache[cache_key]
        raw, path, _ = client.document(cik, candidate_accession, filing)
        result = (extract_trading_symbols(raw), path, sha256(path) if path.is_file() else "")
        client.evidence_cache[cache_key] = result
        return result

    event_filing = by_accession.get(accession, {"accessionNumber": accession, "form": row.get("form", "")})
    try:
        evidence, path, digest = inspect(accession, event_filing)
    except Exception as error:
        evidence, path, digest = [], Path(), ""
        base["resolution_error"] = f"EVENT_FILING:{type(error).__name__}"
    symbol, ambiguous = choose_symbol(evidence, str(row.ticker))
    if symbol:
        return {
            **base,
            "resolved_ticker": symbol,
            "ticker_candidates": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            "resolution_method": "EXACT_EVENT_FILING_EXPLICIT_TRADING_SYMBOL",
            "resolution_confidence": "A",
            "evidence_file": str(path.relative_to(ROOT)) if path.is_file() else "",
            "evidence_sha256": digest,
            "evidence_accession": accession,
            "evidence_filing_time_utc": str(filing_time(event_filing) or ""),
            "resolution_error": "",
        }
    if ambiguous:
        base.update({"ticker_candidates": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                     "resolution_method": "AMBIGUOUS_EVENT_FILING_SYMBOLS",
                     "resolution_error": "AMBIGUOUS"})
        return base
    form = str(row.get("form", "")).upper()
    if form.startswith(("S-1", "F-1")):
        # If the registration filing itself has no explicit listed common
        # symbol, nearby pre-listing registrations cannot establish an
        # event-date traded security. Fail closed without scanning large
        # prospectus histories.
        base["resolution_method"] = "REGISTRATION_FILING_WITHOUT_EVENT_DATE_LISTING_EVIDENCE"
        base["resolution_error"] = "NOT_LISTED_AT_EVENT_DATE_OR_UNPROVEN"
        return base
    event_time = pd.Timestamp(row.event_time_utc)
    event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
    prior = []
    future = []
    for filing in filings:
        stamp = filing_time(filing)
        if stamp is None or str(filing.get("accessionNumber")) == accession:
            continue
        delta = (event_time - stamp).total_seconds() / 86400.0
        if 0 <= delta <= 365:
            prior.append((delta, filing))
        elif -90 <= delta < 0:
            future.append((-delta, filing))
    for delta, filing in sorted(prior, key=lambda item: item[0])[:8]:
        candidate_accession = str(filing.get("accessionNumber"))
        try:
            evidence, path, digest = inspect(candidate_accession, filing)
        except Exception:
            continue
        symbol, ambiguous = choose_symbol(evidence, str(row.ticker))
        if symbol:
            confidence = "B" if delta <= 90 else "C"
            return {
                **base,
                "resolved_ticker": symbol,
                "ticker_candidates": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                "resolution_method": "NEAREST_PRIOR_SAME_CIK_EXPLICIT_TRADING_SYMBOL",
                "resolution_confidence": confidence,
                "evidence_file": str(path.relative_to(ROOT)) if path.is_file() else "",
                "evidence_sha256": digest,
                "evidence_accession": candidate_accession,
                "evidence_filing_time_utc": str(filing_time(filing) or ""),
                "resolution_error": "" if confidence == "B" else "PRIOR_EVIDENCE_OLDER_THAN_90_DAYS",
            }
        if ambiguous:
            base["ticker_candidates"] = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    # Later filings can clarify entity identity, but are never exact-backfill
    # authority and never become model features.
    for _, filing in sorted(future, key=lambda item: item[0])[:4]:
        candidate_accession = str(filing.get("accessionNumber"))
        try:
            evidence, path, digest = inspect(candidate_accession, filing)
        except Exception:
            continue
        symbol, _ = choose_symbol(evidence, str(row.ticker))
        if symbol:
            return {
                **base,
                "resolved_ticker": symbol,
                "ticker_candidates": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                "resolution_method": "NEAREST_FUTURE_SAME_CIK_IDENTITY_ONLY",
                "resolution_confidence": "C",
                "resolution_asof_mode": "POSTHOC_IDENTITY_RESOLUTION_ONLY",
                "evidence_file": str(path.relative_to(ROOT)) if path.is_file() else "",
                "evidence_sha256": digest,
                "evidence_accession": candidate_accession,
                "evidence_filing_time_utc": str(filing_time(filing) or ""),
                "resolution_error": "FUTURE_METADATA_NOT_EXACT_ELIGIBLE",
            }
    current_tickers = [normalize_symbol(value) for value in submission.get("tickers", [])]
    current_tickers = [value for value in current_tickers if value]
    base["ticker_candidates"] = json.dumps(
        [{"symbol": value, "method": "CURRENT_SUBMISSIONS_METADATA", "score": 0}
         for value in current_tickers],
        ensure_ascii=False,
        sort_keys=True,
    )
    base["resolution_error"] = base["resolution_error"] or "NO_EXPLICIT_EVENT_OR_PRIOR_SYMBOL"
    return base


def build_queue(preserve_previous: bool = True) -> dict[str, Any]:
    roles = pd.read_parquet(ROLE_PATH)
    roles = roles.loc[roles.role_status.astype(str).eq("ACTIVE"),
                      ["event_id", "role", "role_hash", "role_frozen_at", "labels_opened"]]
    if roles.labels_opened.astype(str).str.lower().eq("true").any():
        raise RuntimeError("FROZEN_ROLE_LABEL_GUARD_FAIL")
    events = provider._us_candidates()
    metadata_columns = ["event_id", "form", "headline", "source", "raw_file", "cik", "company", "url"]
    source = pd.read_parquet(US_QUEUE_PATH).merge(roles, on="event_id", how="inner")
    source = source.merge(events[metadata_columns], on="event_id", how="left")
    status = pd.read_parquet(PROVIDER_STATUS_PATH)[
        ["event_id", "selected_price_provider", "exact_contract_status", "alpaca_sip_status"]
    ]
    source = source.merge(status, on="event_id", how="left", suffixes=("", "_provider"))
    source["event_time_utc"] = pd.to_datetime(source.event_time_utc, utc=True)
    source["year"] = source.event_time_utc.dt.year
    source = source[source.source_family.astype(str).eq("US_SEC")].copy()
    nonexact = ~source.exact_contract_status.astype(str).eq("EXACT_TIMING_ELIGIBLE")
    dev = source.role.astype(str).eq("DEV_EXTENSION") & source.year.ge(2016) & nonexact
    final = source.role.astype(str).eq("FINAL_META_RESERVE") & source.year.ge(2016) & nonexact
    source = source.loc[dev | final].copy()
    source["priority"] = np.select(
        [source.role.eq("DEV_EXTENSION") & source.year.isin(TARGET_YEARS),
         source.role.eq("DEV_EXTENSION"),
         source.role.eq("FINAL_META_RESERVE") & source.year.isin(TARGET_YEARS)],
        [0, 1, 2],
        default=3,
    )
    source["candidate_reason"] = np.select(
        [source.priority.eq(0), source.priority.eq(1), source.priority.eq(2)],
        ["DEV_US_SEC_2018_2024", "DEV_US_SEC_OTHER", "FINAL_US_DEFICIT_2018_2024"],
        default="FINAL_US_DEFICIT_OTHER",
    )
    parsed = source.event_id.astype(str).map(parse_event_id)
    source["CIK"] = parsed.map(lambda item: item[0])
    source["accession"] = parsed.map(lambda item: item[1])
    source["issuer"] = source.company.fillna(source.issuer).astype(str)
    source["event_date"] = source.event_time_utc.dt.tz_convert(app.US_TZ).dt.date.astype(str)
    defaults: dict[str, Any] = {
        "resolved_ticker": "", "ticker_candidates": "[]", "resolution_method": "UNRESOLVED",
        "resolution_confidence": "D", "resolution_asof_mode": "EVENT_DATE_METADATA_ONLY",
        "evidence_file": "", "evidence_sha256": "", "evidence_accession": "",
        "evidence_filing_time_utc": "", "resolution_error": "", "resolution_status": "UNRESOLVED",
        "fetch_status": "NOT_ATTEMPTED", "fetch_error": "", "exact_eligible": False,
        "provider": "", "provider_symbol": "", "raw_file": "", "raw_sha256": "",
        "attempt_count": 0,
        "alpaca_normal_retry_status": "NOT_ATTEMPTED",
        "alpaca_normal_retry_policy_sha256": "",
        "alpaca_normal_retry_attempt_count": 0,
        **consensus_columns(),
    }
    for key, value in defaults.items():
        source[key] = value
    if preserve_previous and QUEUE_PATH.is_file():
        previous = pd.read_parquet(QUEUE_PATH).set_index("event_id")
        for column in defaults:
            if column not in previous.columns:
                continue
            mapped = source.event_id.astype(str).map(previous[column].to_dict())
            source.loc[mapped.notna(), column] = mapped[mapped.notna()].to_numpy()
    columns = [
        "priority", "candidate_reason", "event_id", "CIK", "issuer", "event_date", "event_time_utc",
        "required_start_utc", "required_end_utc", "ticker", "resolved_ticker", "ticker_candidates",
        "role", "role_hash", "role_frozen_at", "source_family", "form", "accession", "headline", "url",
        "status", "last_error", "selected_price_provider", "alpaca_sip_status",
        "resolution_method", "resolution_confidence", "resolution_asof_mode", "resolution_status",
        "evidence_file", "evidence_sha256", "evidence_accession", "evidence_filing_time_utc",
        "resolution_error", "fetch_status", "fetch_error", "exact_eligible", "provider",
        "provider_symbol", "raw_file", "raw_sha256", "attempt_count",
        "alpaca_normal_retry_status", "alpaca_normal_retry_policy_sha256",
        "alpaca_normal_retry_attempt_count",
        "corroborating_evidence_files", "corroborating_evidence_sha256",
        "corroborating_accessions", "consensus_evidence_count",
        "consensus_contract_sha256", "consensus_status",
    ]
    source = source.sort_values(["priority", "year", "event_time_utc", "event_id"])[columns]
    atomic_parquet(source, QUEUE_PATH)
    report = {
        "timestamp": now(), "selection_uses_labels": False, "outcome_columns_loaded": False,
        "role_assignment_sha256": sha256(ROLE_PATH), "rows": len(source),
        "CIK_complete": int(source.CIK.astype(str).ne("").sum()),
        "by_reason": source.candidate_reason.value_counts().to_dict(),
        "by_role": source.role.value_counts().to_dict(),
        "by_year": pd.to_datetime(source.event_time_utc, utc=True).dt.year.value_counts().sort_index().to_dict(),
        "queue_sha256": sha256(QUEUE_PATH),
    }
    return report


def write_aliases_and_timeline(queue: pd.DataFrame) -> None:
    queue = ensure_consensus_columns(queue)
    resolved = queue.loc[queue.resolved_ticker.astype(str).ne("")].copy()
    alias_columns = ["CIK", "issuer", "event_date", "ticker", "resolved_ticker",
                     "resolution_method", "resolution_confidence", "resolution_asof_mode",
                     "accession", "evidence_file", "evidence_sha256", "role", "event_id",
                     "corroborating_evidence_files", "corroborating_evidence_sha256",
                     "corroborating_accessions", "consensus_evidence_count",
                     "consensus_contract_sha256", "consensus_status"]
    aliases = resolved[alias_columns].rename(columns={"ticker": "queue_ticker"})
    aliases["valid_from"] = aliases.event_date
    aliases["valid_to"] = ""
    atomic_parquet(aliases, ALIAS_PATH)
    if resolved.empty:
        timeline = pd.DataFrame(columns=[
            "CIK", "issuer", "ticker", "resolved_ticker", "valid_from", "valid_to",
            "event_count", "evidence_count", "evidence_methods", "best_confidence",
            "resolution_asof_modes", "validity_semantics",
        ])
    else:
        ranking = {"A": 0, "B": 1, "C": 2, "D": 3}
        timeline_rows = []
        for (cik, symbol), group in resolved.groupby(["CIK", "resolved_ticker"]):
            dates = pd.to_datetime(group.event_date)
            confidence = min(group.resolution_confidence.astype(str), key=lambda value: ranking.get(value, 9))
            consensus_counts = pd.to_numeric(
                group.consensus_evidence_count, errors="coerce"
            ).fillna(0).astype(int)
            evidence_count = int(sum(max(1, value) for value in consensus_counts))
            timeline_rows.append({
                "CIK": cik,
                "issuer": " | ".join(sorted(set(group.issuer.astype(str))))[:1000],
                "ticker": symbol,
                "resolved_ticker": symbol,
                "valid_from": str(dates.min().date()),
                "valid_to": str(dates.max().date()),
                "event_count": len(group),
                "evidence_count": evidence_count,
                "evidence_methods": json.dumps(
                    sorted(set(group.resolution_method.astype(str))), ensure_ascii=False
                ),
                "best_confidence": confidence,
                "resolution_asof_modes": json.dumps(
                    sorted(set(group.resolution_asof_mode.astype(str))), ensure_ascii=False
                ),
                "validity_semantics": "OBSERVED_EVENT_DATE_RANGE_NOT_CONTINUOUS_LISTING_CLAIM",
            })
        timeline = pd.DataFrame(timeline_rows).sort_values(["CIK", "valid_from", "resolved_ticker"])
    atomic_parquet(timeline, TIMELINE_PATH)


def recovery_report(queue: pd.DataFrame, client: SECArchiveClient | None = None) -> dict[str, Any]:
    queue = ensure_consensus_columns(queue)
    years = pd.to_datetime(queue.event_time_utc, utc=True).dt.year
    year_rows = []
    for year in range(2018, 2025):
        subset = queue.loc[years.eq(year)]
        year_rows.append({
            "year": year,
            "queue": len(subset),
            "resolved": int(subset.resolved_ticker.astype(str).ne("").sum()),
            "resolved_new_ticker": int((subset.resolved_ticker.astype(str).ne("") &
                                         subset.resolved_ticker.astype(str).ne(subset.ticker.astype(str))).sum()),
            "bars_found": int(subset.fetch_status.astype(str).isin(
                {"ALPACA_FETCHED_VALID", "TIMING_FAIL"}
            ).sum()),
            "exact_eligible": int(subset.exact_eligible.fillna(False).astype(bool).sum()),
        })
    payload = {
        "schema_version": 1, "timestamp": now(),
        "selection_uses_labels": False, "outcome_columns_loaded": False,
        "events_attempted": int(queue.resolution_status.astype(str).ne("UNRESOLVED").sum()),
        "queue_rows": len(queue),
        "historical_ticker_changed_count": int((queue.resolved_ticker.astype(str).ne("") &
                                                  queue.resolved_ticker.astype(str).ne(queue.ticker.astype(str))).sum()),
        "new_ticker_resolved": int(queue.resolved_ticker.astype(str).ne("").sum()),
        "confidence_counts": queue.resolution_confidence.astype(str).value_counts().to_dict(),
        "resolution_status_counts": queue.resolution_status.astype(str).value_counts().to_dict(),
        "fetch_status_counts": queue.fetch_status.astype(str).value_counts().to_dict(),
        "alpaca_retry_success": int(queue.fetch_status.astype(str).eq("ALPACA_FETCHED_VALID").sum()),
        "new_exact_rows": int(queue.exact_eligible.astype(bool).sum()),
        "exact_by_role": queue.loc[queue.exact_eligible.astype(bool), "role"].value_counts().to_dict(),
        "consensus_status_counts": queue.consensus_status.astype(str).value_counts().to_dict(),
        "multi_filing_consensus_promotions": int(
            queue.resolution_method.astype(str).eq(
                "MULTI_FILING_CONSENSUS_PRIOR_EXPLICIT_TRADING_SYMBOL"
            ).sum()
        ),
        "consensus_policy_sha256": sha256(CONSENSUS_POLICY_PATH)
        if CONSENSUS_POLICY_PATH.is_file() else None,
        "year_distribution": year_rows,
        "role_distribution": queue.role.astype(str).value_counts().to_dict(),
        "queue_sha256": sha256(QUEUE_PATH) if QUEUE_PATH.is_file() else None,
        "alias_registry_sha256": sha256(ALIAS_PATH) if ALIAS_PATH.is_file() else None,
        "timeline_sha256": sha256(TIMELINE_PATH) if TIMELINE_PATH.is_file() else None,
        "SEC_network_requests": client.network_requests if client else 0,
        "SEC_cache_hits": client.cache_hits if client else 0,
    }
    atomic_json(payload, REPORT_PATH)
    return payload


def resolve_queue(max_events: int | None = None, years: set[int] | None = None,
                  min_interval: float = 0.16, workers: int = 1,
                  retry_unresolved: bool = False) -> dict[str, Any]:
    queue = pd.read_parquet(QUEUE_PATH)
    queue_year = pd.to_datetime(queue.event_time_utc, utc=True).dt.year
    selected = queue.resolution_status.astype(str).eq("UNRESOLVED")
    if retry_unresolved:
        selected |= queue.resolution_status.astype(str).isin(
            {"UNRESOLVED_NO_EXPLICIT_SYMBOL", "AMBIGUOUS"}
        )
    if years:
        selected &= queue_year.isin(years)
    indices = list(queue.loc[selected].sort_values(["priority", "event_time_utc"]).index)
    if max_events is not None:
        indices = indices[:max_events]
    client = SECArchiveClient(min_interval=min_interval)

    def apply_result(index: Any, result: dict[str, Any]) -> None:
        for key, value in result.items():
            queue.loc[index, key] = value
        resolved = normalize_symbol(result.get("resolved_ticker"))
        confidence = str(result.get("resolution_confidence", "D"))
        if resolved:
            queue.loc[index, "resolution_status"] = (
                "RESOLVED_SAME_TICKER" if resolved == normalize_symbol(queue.loc[index, "ticker"])
                else "RESOLVED_NEW_TICKER"
            )
            queue.loc[index, "fetch_status"] = (
                "ALPACA_RETRY_PENDING" if confidence in EXACT_CONFIDENCE else "NOT_EXACT_CONFIDENCE"
            )
        elif result.get("resolution_error") == "AMBIGUOUS":
            queue.loc[index, "resolution_status"] = "AMBIGUOUS"
        elif "NOT_LISTED" in str(result.get("resolution_error", "")):
            queue.loc[index, "resolution_status"] = "NOT_LISTED_AT_EVENT_DATE"
        else:
            queue.loc[index, "resolution_status"] = "UNRESOLVED_NO_EXPLICIT_SYMBOL"

    completed = 0
    workers = max(1, int(workers))
    if workers == 1 or len(indices) < 2:
        for index in indices:
            apply_result(index, resolve_one(queue.loc[index], client))
            completed += 1
            if completed % 10 == 0:
                atomic_parquet(queue, QUEUE_PATH)
    else:
        # Resolve all rows for one CIK on the same worker. This avoids cache
        # write races while overlapping SEC network latency. Four workers at
        # >=0.55 seconds/request stay below the SEC 10 requests/second limit.
        grouped: dict[str, list[Any]] = {}
        for index in indices:
            grouped.setdefault(str(queue.loc[index, "CIK"]), []).append(index)
        worker_interval = max(float(min_interval), 0.55)

        def resolve_group(group_indices: list[Any]) -> tuple[list[tuple[Any, dict[str, Any]]], int, int]:
            worker_client = SECArchiveClient(min_interval=worker_interval)
            rows = [(index, resolve_one(queue.loc[index].copy(), worker_client))
                    for index in group_indices]
            return rows, worker_client.network_requests, worker_client.cache_hits

        with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
            ordered_groups = sorted(grouped.values(), key=lambda group: (len(group), group[0]))
            futures = [executor.submit(resolve_group, group) for group in ordered_groups]
            for future in as_completed(futures):
                rows, network_requests, cache_hits = future.result()
                client.network_requests += network_requests
                client.cache_hits += cache_hits
                for index, result in rows:
                    apply_result(index, result)
                    completed += 1
                    if completed % 10 == 0:
                        atomic_parquet(queue, QUEUE_PATH)
    atomic_parquet(queue, QUEUE_PATH)
    write_aliases_and_timeline(queue)
    return recovery_report(queue, client)


def corroborate_queue(
    max_events: int | None = None,
    years: set[int] | None = None,
    roles: set[str] | None = None,
    min_interval: float = 0.16,
    workers: int = 1,
) -> dict[str, Any]:
    """Promote only rows satisfying the preregistered prior-filing contract."""
    policy, policy_sha256 = load_consensus_policy()
    queue = ensure_consensus_columns(pd.read_parquet(QUEUE_PATH))
    queue_year = pd.to_datetime(queue.event_time_utc, utc=True).dt.year
    prior_c = (
        queue.resolution_confidence.astype(str).eq("C")
        & queue.resolution_asof_mode.astype(str).eq("EVENT_DATE_METADATA_ONLY")
    )
    unresolved = queue.resolution_status.astype(str).eq("UNRESOLVED_NO_EXPLICIT_SYMBOL")
    selected = ~queue.exact_eligible.astype(bool) & (prior_c | unresolved)
    selected &= ~queue.form.astype(str).str.upper().str.startswith(("S-1", "F-1"))
    # Never repeat the same label-blind consensus decision under an unchanged
    # frozen contract. A policy hash change is required for re-evaluation.
    selected &= queue.consensus_contract_sha256.astype(str).ne(policy_sha256)
    if years:
        selected &= queue_year.isin(years)
    if roles:
        selected &= queue.role.astype(str).isin(roles)
    indices = list(queue.loc[selected].sort_values(["priority", "event_time_utc", "event_id"]).index)
    if max_events is not None:
        indices = indices[:max_events]
    client = SECArchiveClient(min_interval=min_interval)
    completed = 0
    promoted_indices: list[Any] = []
    status_counts: Counter[str] = Counter()

    def apply_result(index: Any, result: dict[str, Any]) -> None:
        nonlocal completed
        promoted = bool(result.pop("promoted", False))
        status_counts[str(result.get("consensus_status", "UNKNOWN"))] += 1
        for key, value in result.items():
            queue.loc[index, key] = value
        if promoted:
            resolved = normalize_symbol(result.get("resolved_ticker"))
            queue.loc[index, "resolution_status"] = (
                "RESOLVED_SAME_TICKER"
                if resolved == normalize_symbol(queue.loc[index, "ticker"])
                else "RESOLVED_NEW_TICKER"
            )
            queue.loc[index, "fetch_status"] = "ALPACA_RETRY_PENDING"
            queue.loc[index, "fetch_error"] = ""
            queue.loc[index, "exact_eligible"] = False
            queue.loc[index, ["provider", "provider_symbol", "raw_file", "raw_sha256"]] = ""
            promoted_indices.append(index)
        completed += 1

    workers = max(1, int(workers))
    if workers == 1 or len(indices) < 2:
        for index in indices:
            apply_result(index, corroborate_one(queue.loc[index].copy(), client, policy, policy_sha256))
            if completed % 10 == 0:
                atomic_parquet(queue, QUEUE_PATH)
    else:
        grouped: dict[str, list[Any]] = {}
        for index in indices:
            grouped.setdefault(str(queue.loc[index, "CIK"]), []).append(index)
        worker_interval = max(float(min_interval), 0.55)

        def corroborate_group(group_indices: list[Any]) -> tuple[list[tuple[Any, dict[str, Any]]], int, int]:
            worker_client = SECArchiveClient(min_interval=worker_interval)
            rows = [
                (
                    index,
                    corroborate_one(
                        queue.loc[index].copy(), worker_client, policy, policy_sha256
                    ),
                )
                for index in group_indices
            ]
            return rows, worker_client.network_requests, worker_client.cache_hits

        with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
            ordered_groups = sorted(grouped.values(), key=lambda group: (len(group), group[0]))
            futures = [executor.submit(corroborate_group, group) for group in ordered_groups]
            for future in as_completed(futures):
                rows, network_requests, cache_hits = future.result()
                client.network_requests += network_requests
                client.cache_hits += cache_hits
                for index, result in rows:
                    apply_result(index, result)
                    if completed % 10 == 0:
                        atomic_parquet(queue, QUEUE_PATH)
    atomic_parquet(queue, QUEUE_PATH)
    write_aliases_and_timeline(queue)
    recovery = recovery_report(queue, client)
    promoted = queue.loc[promoted_indices].copy() if promoted_indices else queue.iloc[0:0].copy()
    audit = {
        "schema_version": 1,
        "timestamp": now(),
        "policy_path": str(CONSENSUS_POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha256,
        "policy_frozen_before_results": True,
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "provider_availability_used_for_promotion": False,
        "events_attempted": completed,
        "promoted_to_B": len(promoted_indices),
        "consensus_status_counts": dict(status_counts),
        "promoted_by_role": promoted.role.astype(str).value_counts().to_dict(),
        "promoted_by_year": (
            pd.to_datetime(promoted.event_time_utc, utc=True).dt.year.value_counts().sort_index().to_dict()
            if not promoted.empty else {}
        ),
        "promoted_event_ids_sha256": hashlib.sha256(
            "\n".join(sorted(promoted.event_id.astype(str))).encode("utf-8")
        ).hexdigest(),
        "SEC_network_requests": client.network_requests,
        "SEC_cache_hits": client.cache_hits,
        "queue_sha256": recovery["queue_sha256"],
        "exact_contract_relaxed": False,
    }
    atomic_json(audit, CONSENSUS_AUDIT_PATH)
    return audit


def revalidate_parser_results() -> dict[str, Any]:
    """Re-apply current share-class parsing to parser-sensitive A/B rows."""
    queue = pd.read_parquet(QUEUE_PATH)
    changed: list[dict[str, Any]] = []
    for index, row in queue.loc[
        queue.resolution_confidence.astype(str).isin(EXACT_CONFIDENCE)
        & queue.evidence_file.astype(str).ne("")
    ].iterrows():
        try:
            old_evidence = json.loads(str(row.ticker_candidates) or "[]")
        except json.JSONDecodeError:
            old_evidence = []
        methods = {str(item.get("method", "")) for item in old_evidence}
        parser_sensitive = (
            "SEC_PAIRED_SECURITY_SYMBOL_TEXT" in methods
            or "SEC_COVER_TRADING_SYMBOL_TABLE" in methods
            or bool(re.search(r"(?:WS|WT|W|U)$", str(row.resolved_ticker), re.I))
            or bool(re.search(r"(?:WS|W)$", str(row.ticker), re.I))
        )
        if not parser_sensitive:
            continue
        path = ROOT / str(row.evidence_file)
        if not path.is_file():
            continue
        evidence = extract_trading_symbols(path.read_text(encoding="utf-8", errors="replace"))
        symbol, ambiguous = choose_symbol(evidence, str(row.ticker))
        if symbol == str(row.resolved_ticker) and not ambiguous:
            continue
        if bool(row.exact_eligible):
            raise RuntimeError(f"REVALIDATION_WOULD_REVOKE_EXACT:{row.event_id}")
        changed.append({
            "event_id": str(row.event_id), "old_symbol": str(row.resolved_ticker),
            "new_symbol": symbol, "ambiguous": bool(ambiguous),
        })
        queue.loc[index, "ticker_candidates"] = json.dumps(
            evidence, ensure_ascii=False, sort_keys=True
        )
        queue.loc[index, ["fetch_error", "provider", "provider_symbol", "raw_file", "raw_sha256"]] = ""
        queue.loc[index, "exact_eligible"] = False
        if symbol and not ambiguous:
            queue.loc[index, "resolved_ticker"] = symbol
            queue.loc[index, "resolution_status"] = (
                "RESOLVED_SAME_TICKER"
                if symbol == normalize_symbol(row.ticker) else "RESOLVED_NEW_TICKER"
            )
            queue.loc[index, "fetch_status"] = "ALPACA_RETRY_PENDING"
            queue.loc[index, "resolution_error"] = ""
        else:
            queue.loc[index, "resolved_ticker"] = ""
            queue.loc[index, "resolution_status"] = "UNRESOLVED"
            queue.loc[index, "fetch_status"] = "NOT_ATTEMPTED"
            queue.loc[index, "resolution_confidence"] = "D"
            queue.loc[index, "resolution_method"] = "UNRESOLVED"
            queue.loc[index, "resolution_error"] = "PARSER_REVALIDATION_REQUIRED"
    atomic_parquet(queue, QUEUE_PATH)
    write_aliases_and_timeline(queue)
    report = recovery_report(queue)
    return {
        "rows_changed": len(changed),
        "rows_requeued_for_full_resolution": sum(not row["new_symbol"] for row in changed),
        "changes": changed,
        "queue_sha256": report["queue_sha256"],
    }


def update_provider_status(queue: pd.DataFrame) -> None:
    status = pd.read_parquet(PROVIDER_STATUS_PATH).set_index("event_id")
    for column, default in (
        ("provider_symbol", ""), ("symbol_resolution_method", ""),
        ("symbol_resolution_confidence", ""), ("resolution_asof_mode", ""),
    ):
        if column not in status.columns:
            status[column] = default
    indexed = queue.set_index("event_id")
    exact_ids = indexed.index[indexed.exact_eligible.astype(bool)]
    common = status.index.intersection(exact_ids)
    status.loc[common, "alpaca_sip_status"] = "FETCHED_VALID"
    status.loc[common, "selected_price_provider"] = "ALPACA_SIP"
    status.loc[common, "provider_parity_eligible"] = True
    status.loc[common, "exact_contract_status"] = "EXACT_TIMING_ELIGIBLE"
    status.loc[common, "provider_symbol"] = indexed.loc[common, "resolved_ticker"].astype(str)
    status.loc[common, "symbol_resolution_method"] = indexed.loc[common, "resolution_method"].astype(str)
    status.loc[common, "symbol_resolution_confidence"] = indexed.loc[common, "resolution_confidence"].astype(str)
    status.loc[common, "resolution_asof_mode"] = indexed.loc[common, "resolution_asof_mode"].astype(str)
    atomic_parquet(status.reset_index(), PROVIDER_STATUS_PATH)


def append_ledger(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.is_file() else pd.DataFrame(columns=new.columns)
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    atomic_csv(combined, LEDGER_PATH)


def backfill_resolved(max_requests: int | None = None) -> dict[str, Any]:
    refresh_user_environment(ALPACA_ENV_NAMES)
    allowed, reason = alpaca_backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    queue = pd.read_parquet(QUEUE_PATH)
    pending = queue.fetch_status.astype(str).eq("ALPACA_RETRY_PENDING")
    pending &= queue.resolution_confidence.astype(str).isin(EXACT_CONFIDENCE)
    pending &= queue.resolution_asof_mode.astype(str).eq("EVENT_DATE_METADATA_ONLY")
    candidates = queue.loc[pending].sort_values(["priority", "event_time_utc"])
    client = alpaca.AlpacaClient()
    requests_used = 0
    ledger: list[dict[str, Any]] = []
    for (symbol, date), group in candidates.groupby(["resolved_ticker", "event_date"], sort=False):
        if max_requests is not None and requests_used >= max_requests:
            break
        indices = group.index
        cache_path = provider._cache_path(provider.ALPACA_CACHE / "sip", str(symbol), date)
        cached = cache_path.is_file()
        bars = pd.DataFrame()
        status: dict[str, Any] = {"ok": True, "requests": 0}
        if cached:
            try:
                bars = pd.read_parquet(cache_path)
            except Exception:
                cached = False
        if not cached:
            start = pd.to_datetime(group.required_start_utc, utc=True).min()
            end = pd.to_datetime(group.required_end_utc, utc=True).max()
            bars, status = client.get_bars(str(symbol), start, end, "sip", asof=str(date))
            requests_used += int(status.get("requests", 0))
            if status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, cache_path)
        qa_ok = bool(status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"))
        raw_hash = sha256(cache_path) if qa_ok and cache_path.is_file() else ""
        for index in indices:
            queue.loc[index, "attempt_count"] = int(queue.loc[index, "attempt_count"]) + (0 if cached else 1)
            queue.loc[index, "provider"] = "ALPACA_SIP"
            queue.loc[index, "provider_symbol"] = str(symbol)
            queue.loc[index, "raw_file"] = str(cache_path.relative_to(ROOT)) if qa_ok else ""
            queue.loc[index, "raw_sha256"] = raw_hash
            if not qa_ok:
                queue.loc[index, "fetch_status"] = "ALPACA_NO_BAR"
                queue.loc[index, "fetch_error"] = str(status.get("error") or "NO_BAR_OR_QA_FAIL")[:240]
                queue.loc[index, "exact_eligible"] = False
            else:
                exact, timing_reason, _ = alpaca_backfill.timing_only_eligibility(
                    pd.Timestamp(queue.loc[index, "event_time_utc"]), bars["timestamp"]
                )
                queue.loc[index, "fetch_status"] = "ALPACA_FETCHED_VALID" if exact else "TIMING_FAIL"
                queue.loc[index, "fetch_error"] = "" if exact else timing_reason
                queue.loc[index, "exact_eligible"] = bool(exact)
            ledger.append({
                "timestamp": now(), "event_id": queue.loc[index, "event_id"], "CIK": queue.loc[index, "CIK"],
                "event_date": date, "queue_ticker": queue.loc[index, "ticker"], "resolved_ticker": symbol,
                "resolution_method": queue.loc[index, "resolution_method"],
                "confidence": queue.loc[index, "resolution_confidence"], "provider": "ALPACA_SIP",
                "fetch_result": queue.loc[index, "fetch_status"],
                "exact_eligible": bool(queue.loc[index, "exact_eligible"]),
                "role": queue.loc[index, "role"], "source_family": queue.loc[index, "source_family"],
            })
        atomic_parquet(queue, QUEUE_PATH)
        if status.get("http_status") == 401:
            break
    append_ledger(ledger)
    update_provider_status(queue)
    write_aliases_and_timeline(queue)
    report = recovery_report(queue)
    report["alpaca_network_requests_this_run"] = requests_used
    atomic_json(report, REPORT_PATH)
    return report


def ensure_alpaca_normal_retry_columns(queue: pd.DataFrame) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "alpaca_normal_retry_status": "NOT_ATTEMPTED",
        "alpaca_normal_retry_policy_sha256": "",
        "alpaca_normal_retry_attempt_count": 0,
    }
    for column, default in defaults.items():
        if column not in queue.columns:
            queue[column] = default
    return queue


def load_alpaca_normal_retry_policy() -> tuple[dict[str, Any], str]:
    if not ALPACA_NORMAL_RETRY_POLICY_PATH.is_file():
        raise RuntimeError("ALPACA_NORMAL_RETRY_POLICY_MISSING")
    policy = json.loads(ALPACA_NORMAL_RETRY_POLICY_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "policy_id": "ALPACA_RESOLVED_SYMBOL_NORMAL_RETRY_V1",
        "frozen_before_outcome_inspection": True,
        "selection_uses_labels": False,
        "outcome_columns_permitted": False,
        "provider": "ALPACA",
        "feed": "SIP",
        "request_mode": "NO_ASOF_PARAMETER",
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("ALPACA_NORMAL_RETRY_POLICY_GUARD_FAIL")
    return policy, sha256(ALPACA_NORMAL_RETRY_POLICY_PATH)


def retry_no_bar_without_asof(
    max_requests: int | None = None,
    years: set[int] | None = None,
) -> dict[str, Any]:
    """Try a resolved historical symbol once without Alpaca's ``asof`` mapping.

    The earlier historical resolver request used ``asof=event_date``.  This is
    a distinct, preregistered request mode and is limited to audited A/B DEV
    rows.  No outcome or label data is loaded and strict timing is unchanged.
    """
    policy, policy_hash = load_alpaca_normal_retry_policy()
    allowed, reason = alpaca_backfill.authorized_feed("sip")
    if not allowed:
        raise RuntimeError(reason)
    if not NO_BAR_CLASSIFICATION_PATH.is_file():
        raise RuntimeError("HISTORICAL_NO_BAR_CLASSIFICATION_MISSING")
    refresh_user_environment(ALPACA_ENV_NAMES)
    queue = ensure_alpaca_normal_retry_columns(pd.read_parquet(QUEUE_PATH))
    audit = pd.read_parquet(NO_BAR_CLASSIFICATION_PATH)
    if bool(audit.selection_uses_labels.astype(bool).any()):
        raise RuntimeError("NO_BAR_AUDIT_LABEL_GUARD_FAIL")
    eligible_ids = set(
        audit.loc[audit.high_confidence_fallback_candidate.astype(bool), "event_id"].astype(str)
    )
    event_year = pd.to_datetime(queue.event_time_utc, utc=True).dt.year
    pending = queue.event_id.astype(str).isin(eligible_ids)
    pending &= queue.role.astype(str).eq("DEV_EXTENSION")
    pending &= queue.source_family.astype(str).eq("US_SEC")
    pending &= queue.fetch_status.astype(str).eq("ALPACA_NO_BAR")
    pending &= queue.resolution_confidence.astype(str).isin(EXACT_CONFIDENCE)
    pending &= queue.resolution_asof_mode.astype(str).eq("EVENT_DATE_METADATA_ONLY")
    pending &= queue.resolved_ticker.astype(str).ne("")
    # A row is never retried twice, even if the policy file is later altered.
    pending &= pd.to_numeric(
        queue.alpaca_normal_retry_attempt_count, errors="coerce"
    ).fillna(0).eq(0)
    if years is not None:
        pending &= event_year.isin(years)
    priority_years = [int(value) for value in policy["priority_years"]]
    year_rank = {year: rank for rank, year in enumerate(priority_years)}
    candidates = queue.loc[pending].copy()
    candidates["_year_rank"] = event_year.loc[candidates.index].map(
        lambda value: year_rank.get(int(value), len(year_rank))
    )
    candidates = candidates.sort_values(["_year_rank", "event_time_utc", "event_id"])
    client = alpaca.AlpacaClient()
    requests_used = 0
    symbol_dates_attempted = 0
    ledger: list[dict[str, Any]] = []
    result_counts: Counter[str] = Counter()
    for (symbol, date), group in candidates.groupby(["resolved_ticker", "event_date"], sort=False):
        if max_requests is not None and requests_used >= max_requests:
            break
        indices = group.index
        cache_path = provider._cache_path(provider.ALPACA_CACHE / "sip", str(symbol), date)
        cached = cache_path.is_file()
        bars = pd.DataFrame()
        status: dict[str, Any] = {"ok": True, "requests": 0}
        if cached:
            try:
                bars = pd.read_parquet(cache_path)
            except Exception:
                cached = False
        if not cached:
            start = pd.to_datetime(group.required_start_utc, utc=True).min()
            end = pd.to_datetime(group.required_end_utc, utc=True).max()
            # Deliberately omit asof: the prior request already tested that mode.
            bars, status = client.get_bars(str(symbol), start, end, "sip")
            requests_used += int(status.get("requests", 0))
            symbol_dates_attempted += 1
            if status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"):
                alpaca.atomic_parquet(bars, cache_path)
        qa_ok = bool(status.get("ok") and not bars.empty and alpaca.bar_qa(bars).get("pass"))
        raw_hash = sha256(cache_path) if qa_ok and cache_path.is_file() else ""
        for index in indices:
            queue.loc[index, "alpaca_normal_retry_attempt_count"] = 1
            queue.loc[index, "alpaca_normal_retry_policy_sha256"] = policy_hash
            queue.loc[index, "provider"] = "ALPACA_SIP"
            queue.loc[index, "provider_symbol"] = str(symbol)
            queue.loc[index, "attempt_count"] = int(queue.loc[index, "attempt_count"]) + (0 if cached else 1)
            queue.loc[index, "raw_file"] = str(cache_path.relative_to(ROOT)) if qa_ok else ""
            queue.loc[index, "raw_sha256"] = raw_hash
            if not qa_ok:
                retry_status = "NO_BAR"
                queue.loc[index, "fetch_status"] = "ALPACA_NO_BAR"
                queue.loc[index, "fetch_error"] = str(status.get("error") or "NO_BAR_OR_QA_FAIL")[:240]
                queue.loc[index, "exact_eligible"] = False
            else:
                exact, timing_reason, _ = alpaca_backfill.timing_only_eligibility(
                    pd.Timestamp(queue.loc[index, "event_time_utc"]), bars["timestamp"]
                )
                retry_status = "EXACT_TIMING_ELIGIBLE" if exact else "TIMING_FAIL"
                queue.loc[index, "fetch_status"] = "ALPACA_FETCHED_VALID" if exact else "TIMING_FAIL"
                queue.loc[index, "fetch_error"] = "" if exact else timing_reason
                queue.loc[index, "exact_eligible"] = bool(exact)
            queue.loc[index, "alpaca_normal_retry_status"] = retry_status
            result_counts[retry_status] += 1
            ledger.append({
                "timestamp": now(), "event_id": queue.loc[index, "event_id"],
                "CIK": queue.loc[index, "CIK"], "event_date": date,
                "queue_ticker": queue.loc[index, "ticker"], "resolved_ticker": symbol,
                "resolution_method": queue.loc[index, "resolution_method"],
                "confidence": queue.loc[index, "resolution_confidence"],
                "provider": "ALPACA_SIP_NORMAL_RESOLVED_SYMBOL",
                "fetch_result": queue.loc[index, "fetch_status"],
                "exact_eligible": bool(queue.loc[index, "exact_eligible"]),
                "role": queue.loc[index, "role"],
                "source_family": queue.loc[index, "source_family"],
            })
        atomic_parquet(queue, QUEUE_PATH)
        if status.get("http_status") == 401:
            break
    append_ledger(ledger)
    update_provider_status(queue)
    write_aliases_and_timeline(queue)
    report = {
        "timestamp": now(),
        "schema_version": 1,
        "policy_path": str(ALPACA_NORMAL_RETRY_POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "provider_parity_path": str(alpaca_backfill.PARITY_PATH.relative_to(ROOT)),
        "provider_parity_sha256": sha256(alpaca_backfill.PARITY_PATH),
        "selection_uses_labels": False,
        "outcome_columns_loaded": False,
        "request_mode": "NO_ASOF_PARAMETER",
        "candidate_rows_before_budget": int(len(candidates)),
        "candidate_symbol_dates_before_budget": int(
            candidates[["resolved_ticker", "event_date"]].drop_duplicates().shape[0]
        ),
        "symbol_dates_attempted": int(symbol_dates_attempted),
        "network_requests_used": int(requests_used),
        "result_counts": dict(result_counts),
        "exact_rows_added": int(result_counts["EXACT_TIMING_ELIGIBLE"]),
        "queue_sha256": sha256(QUEUE_PATH),
    }
    atomic_json(report, ALPACA_NORMAL_RETRY_REPORT_PATH)
    refreshed = recovery_report(queue)
    refreshed["alpaca_normal_retry"] = report
    atomic_json(refreshed, REPORT_PATH)
    return report


def parse_years(value: str | None) -> set[int] | None:
    if not value:
        return None
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-queue", action="store_true")
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--revalidate", action="store_true")
    parser.add_argument("--retry-unresolved", action="store_true")
    parser.add_argument("--corroborate-consensus", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--retry-no-bar-normal", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--years")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--sec-min-interval", type=float, default=0.16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--consensus-roles", default="DEV_EXTENSION")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output: dict[str, Any] = {}
    if args.build_queue:
        output["queue"] = build_queue(preserve_previous=not args.reset)
    if args.resolve:
        if not QUEUE_PATH.is_file():
            output["queue"] = build_queue()
        output["resolution"] = resolve_queue(
            args.max_events, parse_years(args.years), args.sec_min_interval, args.workers,
            args.retry_unresolved,
        )
    if args.revalidate:
        output["revalidation"] = revalidate_parser_results()
    if args.corroborate_consensus:
        if not QUEUE_PATH.is_file():
            output["queue"] = build_queue()
        roles = {value.strip() for value in args.consensus_roles.split(",") if value.strip()}
        output["consensus"] = corroborate_queue(
            args.max_events, parse_years(args.years), roles,
            args.sec_min_interval, args.workers,
        )
    if args.backfill:
        output["backfill"] = backfill_resolved(args.max_requests)
    if args.retry_no_bar_normal:
        output["alpaca_normal_retry"] = retry_no_bar_without_asof(
            args.max_requests, parse_years(args.years)
        )
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
