"""Hydrate the frozen V221 US_SEC DEV extension with exact-accession text.

This is a label-blind data recovery step.  It reads only event identity and
filing metadata from the frozen extension, downloads the filing's primary
document plus EX-99 exhibits from the same SEC accession, and writes a
separate immutable cache/manifest.  It never edits the frozen epoch CSV.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
from bs4 import BeautifulSoup

from historical_symbol_resolver import ARCHIVE, SECArchiveClient, parse_event_id


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
SOURCE = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"
CACHE = ROOT / "cache" / "v221_epoch_sec_text"
MANIFEST = RESEARCH / "V221_US_SEC_EXACT_ACCESSION_TEXT_MANIFEST.csv"
STATUS = RESEARCH / "V221_US_SEC_EXACT_ACCESSION_TEXT_STATUS.json"
EXPECTED_SOURCE_SHA256 = "e11c60ef38c5b65e676bab79c96cd21e44fdcb6d604e02a550ffc49003ed311b"
MAX_EXHIBITS = 8
MAX_DOCUMENT_BYTES = 25_000_000


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    atomic_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        path,
    )


def plain_text(raw: str) -> str:
    if not raw.strip():
        return ""
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def exhibit_candidates(client: SECArchiveClient, cik: str, accession: str) -> list[dict[str, Any]]:
    try:
        items = (client.archive_index(cik, accession).get("directory", {}) or {}).get("item", []) or []
    except Exception:
        return []
    result = []
    for item in items:
        name = str(item.get("name") or "").strip()
        kind = str(item.get("type") or "").upper().strip()
        lower = name.lower()
        if not name or not re.search(r"\.(?:htm|html|txt)$", lower):
            continue
        is_ex99 = kind.startswith("EX-99") or bool(re.search(r"(?:^|[-_])ex(?:hibit)?[-_]?99", lower))
        if not is_ex99:
            continue
        size = int(item.get("size") or 0)
        if size and size > MAX_DOCUMENT_BYTES:
            continue
        result.append({"name": name, "type": kind, "size": size})
    result.sort(key=lambda row: (row["type"], row["name"]))
    return result[:MAX_EXHIBITS]


def fetch_named_document(client: SECArchiveClient, cik: str, accession: str, name: str) -> tuple[str, Path]:
    compact = re.sub(r"[^0-9]", "", accession)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    path = ROOT / "cache" / "historical_symbol_resolver" / "documents" / f"{int(cik)}_{compact}_{safe}"
    raw = client.get(
        ARCHIVE.format(cik=int(cik), accession=compact, document=quote(name)),
        path,
    )
    return raw.decode("utf-8", errors="replace"), path


def main() -> None:
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"FROZEN_EXTENSION_HASH_MISMATCH:{source_hash}")
    # Deliberately exclude labels, returns, prices, roles and model features.
    frame = pd.read_csv(
        SOURCE,
        compression="gzip",
        usecols=["event_id", "source_family", "event_time_utc", "form"],
        dtype=str,
    )
    frame = frame[frame.source_family.eq("US_SEC")].drop_duplicates("event_id").copy()
    client = SECArchiveClient(min_interval=0.16)
    records: list[dict[str, Any]] = []
    for number, row in enumerate(frame.itertuples(index=False), start=1):
        event_id = str(row.event_id)
        cik, accession = parse_event_id(event_id)
        record: dict[str, Any] = {
            "event_id": event_id,
            "cik": cik,
            "accession": accession,
            "form": str(row.form),
            "event_time_utc": str(row.event_time_utc),
            "status": "ERROR",
            "primary_document": "",
            "exhibit_count": 0,
            "text_path": "",
            "text_sha256": "",
            "text_bytes": 0,
            "raw_document_hashes": "[]",
            "error": "",
        }
        try:
            _, filings = client.filing_rows(cik)
            filing = next((item for item in filings if str(item.get("accessionNumber")) == accession), None)
            if filing is None:
                raise RuntimeError("EXACT_ACCESSION_NOT_IN_SEC_SUBMISSIONS")
            primary_raw, primary_path, primary_name = client.document(cik, accession, filing)
            if not primary_raw:
                raise RuntimeError("PRIMARY_DOCUMENT_EMPTY")
            parts = [f"ACCESSION {accession}\nPRIMARY-DOCUMENT {primary_name}\n{plain_text(primary_raw)}"]
            raw_hashes = [{
                "kind": "PRIMARY",
                "name": primary_name,
                "path": str(primary_path.relative_to(ROOT)),
                "sha256": sha256(primary_path),
                "bytes": primary_path.stat().st_size,
            }]
            exhibits = exhibit_candidates(client, cik, accession)
            for exhibit in exhibits:
                raw, raw_path = fetch_named_document(client, cik, accession, exhibit["name"])
                text = plain_text(raw)
                if not text:
                    continue
                parts.append(f"EXHIBIT-99 {exhibit['type']} {exhibit['name']}\n{text}")
                raw_hashes.append({
                    "kind": exhibit["type"] or "EX-99",
                    "name": exhibit["name"],
                    "path": str(raw_path.relative_to(ROOT)),
                    "sha256": sha256(raw_path),
                    "bytes": raw_path.stat().st_size,
                })
            combined = "\n\n".join(parts).strip() + "\n"
            compact = re.sub(r"[^0-9]", "", accession)
            text_path = CACHE / f"{int(cik)}_{compact}.txt"
            atomic_text(combined, text_path)
            record.update({
                "status": "PASS",
                "primary_document": primary_name,
                "exhibit_count": len(raw_hashes) - 1,
                "text_path": str(text_path.relative_to(ROOT)),
                "text_sha256": sha256(text_path),
                "text_bytes": text_path.stat().st_size,
                "raw_document_hashes": json.dumps(raw_hashes, sort_keys=True),
            })
        except Exception as error:
            record["error"] = f"{type(error).__name__}:{error}"[:500]
        records.append(record)
        if number % 10 == 0 or number == len(frame):
            passed = sum(item["status"] == "PASS" for item in records)
            print(f"[SEC exact text] {number}/{len(frame)} pass={passed}", flush=True)
    manifest = pd.DataFrame(records)
    atomic_csv(manifest, MANIFEST)
    passed = manifest.status.eq("PASS")
    status = {
        "version": "V221_US_SEC_EXACT_ACCESSION_TEXT_V1",
        "created_at": now(),
        "status": "PASS" if bool(passed.all()) else "PARTIAL",
        "selection_policy": "frozen US_SEC event identities; exact accession primary document plus EX-99 exhibits",
        "selection_uses_labels": False,
        "selection_uses_returns": False,
        "selection_uses_prices": False,
        "selection_uses_roles": False,
        "frozen_source": str(SOURCE.relative_to(ROOT)),
        "frozen_source_sha256": source_hash,
        "input_rows": int(len(frame)),
        "pass_rows": int(passed.sum()),
        "failed_rows": int((~passed).sum()),
        "body_coverage": float(passed.mean()),
        "exhibit_rows": int((manifest.exhibit_count.astype(int) > 0).sum()),
        "exhibit_documents": int(manifest.exhibit_count.astype(int).sum()),
        "text_bytes": int(manifest.text_bytes.astype(int).sum()),
        "network_requests": client.network_requests,
        "cache_hits": client.cache_hits,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST),
        "frozen_epoch_modified": False,
    }
    atomic_json(status, STATUS)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
