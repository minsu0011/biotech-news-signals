"""Precommit membership and recover same-accession text before any outcomes."""
from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
import pandas as pd
import numpy as np
import v224_prospective_locked_evaluation as locked
import v224_historical_mini_seal_preaudit as pre
import historical_symbol_resolver as resolver
import hydrate_v221_us_sec_event_documents as hydrate

ROOT = Path(__file__).resolve().parent
OUT = pre.OUT


class IsolatedClient(resolver.SECArchiveClient):
    """Reuse original cache read-only and write new downloads under this certificate."""
    @staticmethod
    def location(cache_path):
        if cache_path.is_file():
            return cache_path
        relative = cache_path.relative_to(ROOT / "cache")
        old = OUT / "text_raw" / relative
        if old.is_file():
            return old
        key = hashlib.sha256(str(relative).encode()).hexdigest()[:32]
        return OUT / "text_raw" / (key + cache_path.suffix)

    def get(self, url, cache_path):
        if cache_path.is_file():
            self.cache_hits += 1
            return cache_path.read_bytes()
        isolated = self.location(cache_path)
        return super().get(url, isolated)


def write_json(name, value):
    return locked.write_immutable(OUT / name, locked.pretty_json_bytes(value))


def main():
    audit = json.loads((OUT / "SEAL_CONTAMINATION_PREAUDIT.json").read_text(encoding="utf-8"))
    assert audit["exposed_candidate_count"] == 0 and not audit["errors"]
    membership = pd.read_csv(OUT / "CANDIDATE_MEMBERSHIP_METADATA.csv", dtype=str, keep_default_na=False)
    assert len(membership) == 60 and membership.labels_opened_metadata.eq("False").all()
    wanted = set(membership.canonical_event_id)
    accession_set = {value.rsplit(":", 1)[-1] for value in wanted}
    # No numeric outcomes are emitted or used: document matches are only audit evidence.
    text_matches, scanned = [], 0
    dirs = [ROOT / "research", ROOT / "state", ROOT / "cert", ROOT / "output", ROOT / "logs"]
    dirs += list(ROOT.glob("output_V*"))
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md", ".log", ".txt"}:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            found = sorted(accession_set & set(re.findall(r"\d{10}-\d{2}-\d{6}", text)))
            if found:
                text_matches.append({"path": str(path.relative_to(ROOT)), "accessions": found})
    write_json("TEXT_LEDGER_PREAUDIT.json", {"files_scanned": scanned, "matches": text_matches,
                                           "outcome_rows_opened": 0})
    if text_matches:
        print(json.dumps({"status": "TEXT_REFERENCES_REQUIRE_REVIEW", "matches": text_matches}), flush=True)
        return
    # Original queue excludes off-hours and all forms outside the V36 allowlist.
    # Recheck the timestamp/session contract, with timestamp-only price projections.
    timings = []
    for row in membership.itertuples(index=False):
        event = pd.Timestamp(row.event_time)
        local, session_open, session_close = pre.provider.app.session_bounds(event, "US")
        assert local.weekday() < 5 and session_open <= local <= session_close
        path = ROOT / row.price_path
        stamps = pd.read_parquet(path, columns=["timestamp"])
        values = pd.to_datetime(stamps.timestamp, utc=True).sort_values().drop_duplicates().array.asi8
        entry = pd.Timestamp(values[np.searchsorted(values, (event + pd.Timedelta(minutes=2)).value)], tz="UTC")
        exit_time = pd.Timestamp(values[np.searchsorted(values, (entry + pd.Timedelta(minutes=30)).value)], tz="UTC")
        assert entry <= session_close.tz_convert("UTC") and exit_time <= session_close.tz_convert("UTC")
        assert ((values >= session_open.tz_convert("UTC").value) &
                (values <= (entry-pd.Timedelta(minutes=1)).value)).sum() >= 2
        timings.append({"canonical_event_id": row.canonical_event_id, "actual_entry_time_utc": entry.isoformat(),
                        "actual_exit_time_utc": exit_time.isoformat(), "regular_session_eligible": True,
                        "exact_contract_eligible": True, "price_columns_read": ["timestamp"]})
    write_json("EXACT_METADATA_CONTRACT_AUDIT.json", {"rows": timings, "passed": True,
                                                     "outcome_values_read": False})
    # Find original feature metadata using a strict column projection.
    event_paths = sorted((ROOT / "data").glob("events*sec*.csv"))
    event_paths += sorted((ROOT / "data/untouched_pool").glob("sec_*.csv.gz"))
    event_metadata = {}
    allowed = {"event_id", "event_time_utc", "headline", "body", "form", "event_type", "source", "url"}
    for path in event_paths:
        header = pd.read_csv(path, nrows=0).columns
        if "event_id" not in header:
            continue
        frame = pd.read_csv(path, usecols=sorted(allowed & set(header)), dtype=str, keep_default_na=False)
        for item in frame.loc[frame.event_id.isin(wanted)].to_dict("records"):
            event_metadata.setdefault(item["event_id"], {**item, "metadata_path": str(path.relative_to(ROOT))})
    if set(event_metadata) != wanted:
        raise RuntimeError("EXACT_EVENT_METADATA_MISSING")
    forms = json.loads((ROOT / "data/SEC_FORM_POLICY_V36.json").read_text(encoding="utf-8"))["allowed_forms"]
    for row in membership.itertuples(index=False):
        item = event_metadata[row.canonical_event_id]
        assert item["form"] in forms
        assert pd.Timestamp(item["event_time_utc"]) == pd.Timestamp(row.event_time)
    write_json("EVENT_METADATA_SOURCE_AUDIT.json", {
        "rows": [{"event_id": k, "path": v["metadata_path"], "form": v["form"]} for k,v in event_metadata.items()],
        "all_forms_allowed": True, "all_event_times_equal": True})
    public_columns = ["cert_row_index", "block_id", "canonical_event_id", "event_time", "ticker", "CIK", "source_family", "role"]
    frozen_members = membership[public_columns].copy()
    frozen_members["role"] = "RESEARCH_SEAL"
    payload = frozen_members.to_csv(index=False, lineterminator="\n").encode()
    digest = locked.write_immutable(OUT / "PRECOMMITTED_MEMBERSHIP.csv", payload)
    locked.write_immutable(OUT / "PRECOMMITTED_MEMBERSHIP_SHA256.txt", (digest + "\n").encode())
    locked.write_immutable(OUT / "CERT_WORKING_STATE_INITIAL.json",
                           (ROOT / locked.FROZEN_REL / locked.FROZEN_STATE_NAME).read_bytes())
    write_json("CONTAMINATION_CLEARANCE.json", {
        "status": "PASS", "tabular_files_audited": audit["files_audited"], "text_documents_audited": scanned,
        "identity_or_accession_overlap": 0, "outcome_accesses": 0,
        "audit_scope": "workspace training/DEV, research/OOF/features/polarity/calibration, prior certs and ledgers",
        "limitation": "No prior exposure found in available file lineage; this is not proof about external/manual access."})
    client = IsolatedClient(min_interval=0.2)
    records = []
    filings_cache = {}
    for number, row in enumerate(membership.itertuples(index=False), 1):
        record = {"event_id": row.canonical_event_id, "status": "FAIL", "outcomes_opened": 0}
        try:
            cik, accession = resolver.parse_event_id(row.canonical_event_id)
            if cik not in filings_cache:
                _, filings_cache[cik] = client.filing_rows(cik)
            filing = next((f for f in filings_cache[cik] if f.get("accessionNumber") == accession), None)
            if filing is None:
                raise RuntimeError("EXACT_ACCESSION_NOT_IN_SUBMISSIONS")
            accepted = pd.Timestamp(filing.get("acceptanceDateTime"))
            if accepted.tzinfo is None:
                raise RuntimeError("AMBIGUOUS_ACCEPTANCE_TIME")
            if accepted != pd.Timestamp(row.event_time):
                raise RuntimeError("EXACT_ACCEPTANCE_TIMESTAMP_MISMATCH")
            if str(filing.get("form")) != event_metadata[row.canonical_event_id]["form"]:
                raise RuntimeError("EXACT_FORM_MISMATCH")
            raw, raw_path, name = client.document(cik, accession, filing)
            if not raw:
                raise RuntimeError("PRIMARY_EMPTY")
            raw_path = client.location(raw_path)
            parts = [f"ACCESSION {accession}\nPRIMARY-DOCUMENT {name}\n{hydrate.plain_text(raw)}"]
            sources = [{"kind": "PRIMARY", "path": str(raw_path.relative_to(ROOT)), "sha256": locked.sha256_file(raw_path)}]
            # Same EX-99 extraction policy as the frozen V221 text preparation.
            for exhibit in hydrate.exhibit_candidates(client, cik, accession):
                text, path = hydrate.fetch_named_document(client, cik, accession, exhibit["name"])
                path = client.location(path)
                plain = hydrate.plain_text(text)
                if not plain:
                    continue
                parts.append(f"EXHIBIT-99 {exhibit['type']} {exhibit['name']}\n{plain}")
                sources.append({"kind": exhibit["type"] or "EX-99", "path": str(path.relative_to(ROOT)), "sha256": locked.sha256_file(path)})
            combined = "\n\n".join(parts).strip() + "\n"
            path = OUT / "text" / f"{int(cik)}_{accession.replace('-', '')}.txt"
            locked.write_immutable(path, combined.encode())
            item = event_metadata[row.canonical_event_id]
            record.update({"status": "PASS", "text_path": str(path.relative_to(ROOT)),
                           "text_sha256": locked.sha256_file(path), "raw_sources": sources,
                           "acceptance_time_utc": accepted.isoformat(), "form": item["form"],
                           "headline": item["headline"], "event_type": item.get("event_type", "OTHER"),
                           "source": item.get("source", "SEC"), "same_accession_only": True})
        except Exception as error:
            record["failure_type"] = type(error).__name__
            record["failure_reason"] = str(error).split(" for url:")[0][:160]
        records.append(record)
        print(json.dumps({"text_rows_processed": number, "passed": sum(r["status"] == "PASS" for r in records),
                          "last_status": record["status"], "failure_reason": record.get("failure_reason"), "outcomes_opened": 0}), flush=True)
    audit_name = "EXACT_ACCESSION_TEXT_AUDIT_RECOVERY.json" if (OUT/"EXACT_ACCESSION_TEXT_AUDIT.json").exists() else "EXACT_ACCESSION_TEXT_AUDIT.json"
    write_json(audit_name, {"rows": records, "network_requests": client.network_requests,
                                                "cache_hits": client.cache_hits, "passed": all(r["status"] == "PASS" for r in records)})


if __name__ == "__main__":
    main()
