"""Compare only event identities against previously opened research artifacts."""
from __future__ import annotations
import csv
import gzip
import json
import re
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cert_research/V224_HISTORICAL_MINI_SEAL"
IDENTITIES = {"event_id", "canonical_event_id", "source_event_id"}
OUTCOMES = {"y", "fwd_ret_30m", "future_return", "correct", "correctness", "strategy_net", "label", "up_down"}


def accession(value):
    match = re.search(r"\d{10}-\d{2}-\d{6}", str(value))
    return match.group(0) if match else str(value)


def main():
    destination = OUT / "SEAL_CONTAMINATION_PREAUDIT.json"
    if destination.exists():
        raise RuntimeError("CONTAMINATION_AUDIT_ALREADY_EXISTS")
    selected = pd.read_csv(OUT / "CANDIDATE_MEMBERSHIP_METADATA.csv", dtype=str)
    wanted = set(selected.canonical_event_id)
    aliases = {accession(value): value for value in wanted}
    paths = set((ROOT / "data").glob("*labeled*.csv*"))
    paths.add(ROOT / "data/V59_DEV_EXTENSION_LABELED.csv.gz")
    directories = [ROOT / "cert", ROOT / "research", ROOT / "output", ROOT / "staging"]
    directories += list(ROOT.glob("output_V*"))
    for directory in directories:
        for path in directory.rglob("*"):
            if path.is_file() and (path.name.endswith((".csv", ".csv.gz", ".parquet"))):
                paths.add(path)
    base = ROOT / "data/HIST_HIGH_CONF_NEWS_BASE_V1"
    paths.update(base.glob("*.parquet"))
    records, overlaps, errors = [], [], []
    for n, path in enumerate(sorted(paths)):
        relative = str(path.relative_to(ROOT))
        try:
            if path.suffix == ".parquet":
                columns = pq.read_schema(path).names
            else:
                columns = list(pd.read_csv(path, nrows=0).columns)
            ids = sorted(IDENTITIES & set(columns))
            targets = sorted(OUTCOMES & set(columns))
            if not ids:
                records.append({"path": relative, "status": "NO_EVENT_ID_COLUMNS", "outcome_column_names": targets})
                continue
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path, columns=ids)
            else:
                frame = pd.read_csv(path, usecols=ids, dtype=str, keep_default_na=False)
            found = set()
            for column in ids:
                values = set(frame[column].dropna().astype(str))
                found.update(wanted & values)
                found.update(aliases[a] for value in values if (a := accession(value)) in aliases)
            record = {"path": relative, "status": "IDENTITY_COLUMNS_ONLY", "rows": len(frame),
                      "columns_read": ids, "outcome_column_names": targets,
                      "overlap_event_ids": sorted(found)}
            records.append(record)
            if found:
                overlaps.append(record)
                print(json.dumps({"overlap_file": relative, "overlap_count": len(found),
                                  "outcome_schema_present": bool(targets)}), flush=True)
        except Exception as error:
            errors.append({"path": relative, "error_type": type(error).__name__})
        if (n + 1) % 100 == 0:
            print(json.dumps({"identity_files_audited": n + 1, "seal_outcome_rows_read": 0}), flush=True)
    exposed = sorted({event for record in overlaps if record["outcome_column_names"]
                      for event in record["overlap_event_ids"]})
    report = {
        "status": "FAIL_CLOSED_SEAL_CONTAMINATION" if exposed else
                  "IDENTITY_AUDIT_ERRORS_REQUIRE_REVIEW" if errors else "NO_IDENTITY_EXPOSURE_FOUND_PENDING_TEXT_LEDGER_AUDIT",
        "selected_rows": len(selected), "files_considered": len(paths),
        "files_audited": len(records), "errors": errors, "overlap_artifacts": overlaps,
        "exposed_candidate_event_ids": exposed, "exposed_candidate_count": len(exposed),
        "audit_projection": "identity columns only; outcome column names inspected as schema",
        "accession_alias_matching": True, "seal_outcome_rows_read": 0,
        "final_meta_outcome_rows_read": 0, "all_file_audits": records,
    }
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key not in
                      {"all_file_audits", "overlap_artifacts", "exposed_candidate_event_ids"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
