"""Fail-closed, one-shot access to exactly precommitted historical outcome IDs."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class SealAccessGuard:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.membership_path = self.directory / "PRECOMMITTED_MEMBERSHIP.csv"
        self.membership_sha = (self.directory / "PRECOMMITTED_MEMBERSHIP_SHA256.txt").read_text().strip()
        if digest(self.membership_path) != self.membership_sha:
            raise RuntimeError("MEMBERSHIP_HASH_MISMATCH")
        self.members = pd.read_csv(self.membership_path, dtype=str)
        if (len(self.members) != 60 or self.members.canonical_event_id.duplicated().any()
                or not self.members.source_family.eq("US_SEC").all()
                or not self.members.role.eq("RESEARCH_SEAL").all()
                or self.members.cert_row_index.astype(int).tolist() != list(range(1,61))
                or self.members.block_id.astype(int).tolist() != [1]*20+[2]*20+[3]*20):
            raise RuntimeError("MEMBERSHIP_SCOPE_INVALID")
        ordered = self.members.assign(_time=pd.to_datetime(self.members.event_time, utc=True)).sort_values(
            ["_time", "canonical_event_id"])
        if ordered.canonical_event_id.tolist() != self.members.canonical_event_id.tolist():
            raise RuntimeError("MEMBERSHIP_NOT_CHRONOLOGICAL")
        self.journal = self.directory / "OUTCOME_ACCESS_JOURNAL.jsonl"
        if self.journal.exists():
            raise RuntimeError("ONE_SHOT_ALREADY_STARTED_NO_AUTOMATIC_REPLAY")
        self.allowed = set()
        self.opened = set()
        self.completed = 0
        self.prediction_path = None
        self.prediction_sha = None

    def authorize(self, block: int):
        if block != self.completed + 1 or not 1 <= block <= 3 or self.allowed:
            raise RuntimeError("BLOCK_ORDER_INVALID")
        if digest(self.membership_path) != self.membership_sha:
            raise RuntimeError("MEMBERSHIP_MUTATED")
        path = self.directory / f"BLOCK_{block:02d}_PREDICTIONS_FROZEN.csv"
        proof = json.loads((self.directory / f"BLOCK_{block:02d}_PREDICTION_FREEZE.json").read_text())
        expected = set(self.members.loc[self.members.block_id.eq(str(block)), "canonical_event_id"])
        frame = pd.read_csv(path, usecols=["canonical_event_id"], dtype=str)
        if len(frame) != 20 or set(frame.canonical_event_id) != expected or frame.canonical_event_id.duplicated().any():
            raise RuntimeError("PREDICTION_MEMBERSHIP_MISMATCH")
        if (proof.get("membership_sha") != self.membership_sha
                or proof.get("prediction_sha") != digest(path)
                or proof.get("deterministic") is not True):
            raise RuntimeError("PREDICTION_FREEZE_INVALID")
        self.allowed = expected
        self.prediction_path = path
        self.prediction_sha = proof["prediction_sha"]

    def open_one(self, event_id: str, reader):
        if event_id not in self.allowed or event_id in self.opened or len(self.opened) >= 60:
            raise RuntimeError("OUTCOME_ACCESS_DENIED")
        if digest(self.membership_path) != self.membership_sha or digest(self.prediction_path) != self.prediction_sha:
            raise RuntimeError("PREOPEN_INTEGRITY_FAILURE")
        # A crash/read error is counted as an attempted opening and cannot be rerun silently.
        record = {"block_id": self.completed + 1, "event_id": event_id, "opened_at": utc_now(),
                  "membership_sha": self.membership_sha, "prediction_sha": self.prediction_sha,
                  "phase": "DURABLE_INTENT_BEFORE_OUTCOME_READ"}
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.opened.add(event_id)
        return reader(event_id)

    def finish_block(self, outcome_columns):
        if not self.allowed or not self.allowed.issubset(self.opened):
            raise RuntimeError("BLOCK_OUTCOMES_INCOMPLETE")
        members = self.members.loc[self.members.block_id.eq(str(self.completed + 1))]
        path = self.directory / "SEAL_OPEN_LEDGER.csv"
        record = {"block_id": self.completed + 1, "opened_at": utc_now(), "row_count": 20,
                  "first_event_time": members.event_time.iloc[0], "last_event_time": members.event_time.iloc[-1],
                  "membership_sha": self.membership_sha, "prediction_sha": self.prediction_sha,
                  "outcome_columns_opened": json.dumps(list(outcome_columns)),
                  "opened_row_ids_sha": hashlib.sha256("\n".join(sorted(self.allowed)).encode()).hexdigest()}
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if new:
                writer.writeheader()
            writer.writerow(record)
            handle.flush()
            os.fsync(handle.fileno())
        self.completed += 1
        self.allowed.clear()

    def counters(self):
        return {"unique_outcome_rows_opened": len(self.opened), "completed_blocks": self.completed,
                "FINAL_META_OUTCOME_ROWS_READ": 0, "NON_US_SEC_RESEARCH_SEAL_OUTCOME_ROWS_READ": 0,
                "REMAINING_US_SEC_OUTCOME_ROWS_READ": 0}
