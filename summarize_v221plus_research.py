"""Build the V221+ research-status summary from audited artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "research" / "audits" / "V221PLUS_CONTRIBUTION_POLARITY" / "FULL_OLD_DEV"
OUTPUT = ROOT / "research" / "V221PLUS_RESEARCH_STATUS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    selected = frame.loc[:, [column for column in columns if column in frame]].copy()
    selected = selected.astype(object).where(pd.notna(selected), None)
    return selected.to_dict(orient="records")


def main() -> None:
    classification = pd.read_csv(AUDIT / "FEATURE_REVERSAL_CLASSIFICATION_V221PLUS.csv")
    scorecard = pd.read_csv(AUDIT / "POLARITY_REVERSAL_SCORECARD_V221PLUS.csv")
    reachability = json.loads(
        (ROOT / "research" / "V221_US_SEC_DEV_REACHABILITY_AUDIT.json").read_text(encoding="utf-8")
    )
    merged = classification.merge(
        scorecard,
        on=["source_family", "feature_block", "role"],
        suffixes=("_class", "_score"),
        validate="one_to_one",
    )
    chosen = "chosen_alpha_class"
    stability = "stability_class"
    production = classification.production_material_candidate.eq(True)
    reversal = merged[
        merged.classification.astype(str).str.contains("POLARITY_REVERSED", na=False)
    ]
    confirmed_nonzero = merged[
        merged.outer_fixed_state_confirmed_class.eq(True)
        & merged[chosen].ne(0.0)
        & merged[stability].ge(0.75)
    ]
    confirmed_ablation_direction = merged[
        merged.role.eq("direction")
        & merged[chosen].eq(0.0)
        & merged.outer_fixed_state_confirmed_class.eq(True)
        & merged[stability].ge(0.75)
    ]
    reversal_columns = [
        "source_family", "feature_block", "role", "classification", chosen,
        stability, "support_class", "outer_fixed_state_primary_class",
        "outer_best_basic_control_primary_class", "outer_fixed_state_margin_class",
        "confidence_correctness_auc", "HC_accuracy", "HC_net", "bootstrap_HC_lower",
        "alpha_minus1", "alpha_zero", "alpha_plus1",
    ]
    signal_columns = [
        "source_family", "feature_block", "role", "classification", chosen,
        stability, "outer_fixed_state_margin_class", "AUC", "BA", "edge",
        "confidence_correctness_auc", "HC_accuracy", "HC_net", "bootstrap_HC_lower",
        "material_move_AUC", "magnitude_ranking",
    ]
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OLD_DEV_NESTED_DIAGNOSTIC_COMPLETE_NEW_DATA_EPOCH_BLOCKED",
        "production_intervention_authorized": False,
        "production_material_candidate_count": int(production.sum()),
        "old_dev": {
            "scorecard_rows": int(len(scorecard)),
            "classification_rows": int(len(classification)),
            "stable_nonzero_outer_confirmed": records(confirmed_nonzero, signal_columns),
            "stable_direction_ablation_outer_confirmed": records(
                confirmed_ablation_direction, signal_columns
            ),
            "polarity_reversal_candidates": records(reversal, reversal_columns),
        },
        "new_data": {
            "epoch_ready": False,
            "replication_executed": False,
            "us_sec_dev_exact_now": reachability["current_exact_timing_eligible"],
            "us_sec_dev_floor": reachability["frozen_floor"]["n"],
            "strict_no_new_future_event_upper_bound": reachability[
                "strict_no_new_future_event_upper_bound"
            ],
            "strict_upper_bound_gap": reachability["strict_upper_bound_gap"],
        },
        "research_questions": {
            "1_new_data_high_confidence_reversal": "NOT_TESTABLE_EPOCH_NOT_READY",
            "2_causal_reversal_block": "NONE_ESTABLISHED; US_NEWS return_path confidence was inner-selected -1 but outer not confirmed",
            "3_stable_plus_zero_minus": "Several +1 opportunity signals and two direction ablations are OLD-DEV outer-confirmed; no -1 state is outer-confirmed",
            "4_sign_or_magnitude": "No sign reversal established; extreme/magnitude flags remain diagnostic only",
            "5_direction_plus_confidence_minus": "NONE_ESTABLISHED",
            "6_opportunity_role_divergence": "PRESENT_ON_OLD_DEV_AS_PLUS1_OPPORTUNITY_WITH_ZERO_DIRECTION_OR_CONFIDENCE; NEW DATA pending",
            "7_reversal_aware_confidence_hc_accuracy": "NOT_MATERIAL; the only -1 candidate has outer fixed-state primary margin below controls",
            "8_bootstrap_hc_net_lower_positive": "NO for the -1 confidence candidate",
            "9_direction_pollution": "OLD-DEV candidates: US_NEWS trend_position ablation and KR_NEWS event_lexical_flags ablation; NEW DATA confirmation required",
            "10_old_polarity_survives_new_massive_kis": "NOT_TESTABLE_EPOCH_NOT_READY",
        },
        "invariants": {
            "outer_labels_used_for_selection": False,
            "raw_feature_sign_flip_performed": False,
            "new_data_labels_used": False,
            "research_seal_or_final_opened": False,
            "v69_changed": False,
        },
        "authority": {
            "full_old_dev_manifest_sha256": sha256(AUDIT / "MANIFEST.json"),
            "scorecard_sha256": sha256(AUDIT / "POLARITY_REVERSAL_SCORECARD_V221PLUS.csv"),
            "classification_sha256": sha256(AUDIT / "FEATURE_REVERSAL_CLASSIFICATION_V221PLUS.csv"),
            "reachability_audit_sha256": sha256(
                ROOT / "research" / "V221_US_SEC_DEV_REACHABILITY_AUDIT.json"
            ),
        },
    }
    temporary = OUTPUT.with_name(OUTPUT.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT), "output_sha256": sha256(OUTPUT),
        "production_material_candidate_count": payload["production_material_candidate_count"],
        "epoch_ready": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
