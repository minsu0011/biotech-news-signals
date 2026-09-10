"""Fail-closed repair for event-archive full-submission exclusion."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

import annual_plus_sec_symbol_recovery_v221 as hybrid
import historical_symbol_resolver as resolver


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
POLICY_PATH = DATA / "EVENT_ARCHIVE_FULL_SUBMISSION_EXCLUSION_REPAIR_POLICY_V2.json"
IDENTITY_POLICY_PATH = DATA / "HISTORICAL_EVENT_ACCESSION_ARCHIVE_SYMBOL_POLICY_V1.json"
IDENTITY_AUDIT_PATH = RESEARCH / "HISTORICAL_EVENT_ACCESSION_ARCHIVE_SYMBOL_AUDIT.json"
REPAIR_V1_AUDIT_PATH = RESEARCH / "EVENT_ARCHIVE_FULL_SUBMISSION_EXCLUSION_REPAIR_AUDIT.json"
AUDIT_PATH = RESEARCH / "EVENT_ARCHIVE_FULL_SUBMISSION_EXCLUSION_REPAIR_V2_AUDIT.json"


def load_policy() -> tuple[dict[str, Any], dict[str, Any], str]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8-sig"))
    identity = json.loads(IDENTITY_POLICY_PATH.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version": 2,
        "frozen_before_compliance_reparse_results": True,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "exact_price_contract_unchanged": True,
        "failure_action": "DEMOTE_TO_D_UNRESOLVED_FAIL_CLOSED",
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise RuntimeError("EVENT_ARCHIVE_REPAIR_POLICY_GUARD_FAIL")
    bindings = {
        resolver.QUEUE_PATH: policy["activation_queue_sha256"],
        REPAIR_V1_AUDIT_PATH: policy["activation_repair_v1_audit_sha256"],
        IDENTITY_POLICY_PATH: policy["required_event_archive_policy_sha256"],
    }
    for path, expected in bindings.items():
        if resolver.sha256(path) != expected:
            raise RuntimeError(f"EVENT_ARCHIVE_REPAIR_BINDING_MISMATCH:{path.name}")
    return policy, identity, resolver.sha256(POLICY_PATH)


def main() -> int:
    policy, identity, policy_sha = load_policy()
    queue = pd.read_parquet(resolver.QUEUE_PATH)
    mask = queue.event_archive_symbol_status.astype(str).eq(
        str(policy["required_event_archive_status"])
    )
    mask &= queue.event_archive_symbol_contract_sha256.astype(str).eq(
        str(policy["required_event_archive_policy_sha256"])
    )
    indices = list(queue.loc[mask].index)
    ids = queue.loc[indices, "event_id"].astype(str).tolist()
    digest = hybrid.event_ids_sha256(ids)
    if len(indices) != int(policy["candidate_count"]):
        raise RuntimeError("EVENT_ARCHIVE_REPAIR_CANDIDATE_COUNT_MISMATCH")
    if digest != policy["candidate_event_ids_sha256"]:
        raise RuntimeError("EVENT_ARCHIVE_REPAIR_CANDIDATE_HASH_MISMATCH")
    allowed_methods = {
        str(value) for value in identity["allowed_explicit_symbol_methods"]
    }
    passed = 0
    demoted = 0
    removed_files = 0
    row_results: list[dict[str, Any]] = []
    for index in indices:
        row = queue.loc[index]
        _, accession = resolver.parse_event_id(str(row.event_id))
        compact = re.sub(r"[^0-9]", "", accession)
        excluded_names = {f"{accession.lower()}.txt", f"{compact}.txt"}
        files = json.loads(str(row.event_archive_symbol_evidence_files))
        hashes = json.loads(str(row.event_archive_symbol_evidence_sha256))
        remaining: list[tuple[str, str, list[dict[str, Any]], str]] = []
        for relative, expected_hash in zip(files, hashes):
            basename = Path(relative).name.lower()
            if any(basename.endswith(f"_{name}") for name in excluded_names):
                removed_files += 1
                continue
            path = ROOT / relative
            if not path.is_file() or resolver.sha256(path) != str(expected_hash):
                continue
            evidence = [
                item
                for item in resolver.extract_trading_symbols(
                    path.read_text(encoding="utf-8", errors="replace")
                )
                if str(item.get("method", "")) in allowed_methods
            ]
            symbol, ambiguous = resolver.choose_symbol(evidence, "")
            if symbol and not ambiguous:
                remaining.append((str(relative), str(expected_hash), evidence, symbol))
        symbols = {
            resolver.normalize_symbol(item[3]) for item in remaining
        } - {""}
        original_symbol = resolver.normalize_symbol(row.resolved_ticker)
        compliant = (
            len(remaining) >= int(policy["minimum_remaining_non_submission_documents"])
            and len(symbols) == 1
            and next(iter(symbols), "") == original_symbol
        )
        if compliant:
            passed += 1
            queue.loc[index, "event_archive_symbol_evidence_files"] = json.dumps(
                [item[0] for item in remaining], ensure_ascii=False
            )
            queue.loc[index, "event_archive_symbol_evidence_sha256"] = json.dumps(
                [item[1] for item in remaining], ensure_ascii=False
            )
            queue.loc[index, "event_archive_symbol_evidence_count"] = len(remaining)
            queue.loc[index, "evidence_file"] = remaining[0][0]
            queue.loc[index, "evidence_sha256"] = remaining[0][1]
            queue.loc[index, "ticker_candidates"] = json.dumps(
                remaining[0][2], ensure_ascii=False, sort_keys=True
            )
            result = "PASS_AFTER_FULL_SUBMISSION_EXCLUSION"
        else:
            demoted += 1
            queue.loc[index, "event_archive_symbol_status"] = "POLICY_COMPLIANCE_FAIL_DEMOTED"
            queue.loc[index, "resolution_confidence"] = "D"
            queue.loc[index, "resolution_method"] = "UNRESOLVED"
            queue.loc[index, "resolved_ticker"] = ""
            queue.loc[index, "resolution_status"] = "UNRESOLVED_NO_EXPLICIT_SYMBOL"
            queue.loc[index, "exact_eligible"] = False
            queue.loc[index, "fetch_status"] = "EVENT_ARCHIVE_POLICY_COMPLIANCE_FAIL"
            queue.loc[index, "fetch_error"] = "FULL_SUBMISSION_EXCLUSION_LEFT_NO_UNAMBIGUOUS_SYMBOL"
            result = "FAIL_DEMOTED"
        queue.loc[index, "event_archive_exclusion_repair_status"] = result
        queue.loc[index, "event_archive_exclusion_repair_policy_sha256"] = policy_sha
        row_results.append({"event_id": str(row.event_id), "result": result})
    resolver.atomic_parquet(queue, resolver.QUEUE_PATH)
    resolver.write_aliases_and_timeline(queue)
    recovery = resolver.recovery_report(queue)
    audit = {
        "schema_version": 2,
        "timestamp": resolver.now(),
        "status": "COMPLETE",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha,
        "selected_rows": len(indices),
        "selected_event_ids_sha256": digest,
        "full_submission_files_removed": removed_files,
        "compliance_passed": passed,
        "demoted_fail_closed": demoted,
        "row_results": row_results,
        "selection_uses_labels": False,
        "selection_uses_prices_or_provider_availability": False,
        "outcome_columns_loaded": False,
        "exact_contract_relaxed": False,
        "roles_reassigned": False,
        "seal_or_final_labels_opened": False,
        "queue_sha256": recovery["queue_sha256"],
    }
    resolver.atomic_json(audit, AUDIT_PATH)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
