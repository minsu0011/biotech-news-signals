from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import v224_prospective_locked_evaluation as evaluation


SOURCE_ROOT = Path(__file__).resolve().parent


class FakeFrozenScorer:
    NUMERIC_COLS = ["pre_ret_1"]
    COST = 0.002

    @staticmethod
    def canonical_json_bytes(value: Any) -> bytes:
        return evaluation.canonical_json_bytes(value)

    @staticmethod
    def state_fingerprint(state: dict[str, Any]) -> str:
        return evaluation.sha256_bytes(evaluation.canonical_json_bytes(state))

    @staticmethod
    def score_digest(frame: pd.DataFrame) -> str:
        columns = [
            "event_id", "raw_probability", "oriented_probability", "probability",
            "prediction", "margin", "high_conf", "confidence_signal",
            "polarity_reversed", "direction_cutoff", "confidence_polarity",
            "confidence_threshold", "confirmation_block",
        ]
        normalized = frame[columns].copy()
        for column in ("prediction", "high_conf", "polarity_reversed"):
            normalized[column] = normalized[column].astype(str).str.lower().map({"true": True, "false": False})
        return evaluation.canonical_records_sha256(normalized)

    @staticmethod
    def score_block(features: pd.DataFrame, bundle: Any, state: dict[str, Any]) -> pd.DataFrame:
        assert not set(features.columns).intersection(evaluation.OUTCOME_COLUMNS)
        ordered = features.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
        block = int(state["machine_state"]["next_confirmation_block"])
        cutoff = 0.50 + 0.01 * (block - 1)
        raw = np.linspace(0.35, 0.65, len(ordered))
        probability = np.clip(0.5 + raw - cutoff, 0, 1)
        margin = np.abs(probability - 0.5)
        threshold = float(np.quantile(margin, 0.8))
        output = ordered[["event_id", "event_time_utc", "source_family", "form", "event_type"]].copy()
        output["numeric_probability"] = raw
        output["text_probability"] = raw
        output["semantic_structure_probability"] = raw
        output["raw_probability"] = raw
        output["oriented_probability"] = raw
        output["probability"] = probability
        output["prediction"] = probability >= 0.5
        output["margin"] = margin
        output["high_conf"] = margin >= threshold
        output["confidence_signal"] = margin
        output["polarity_reversed"] = False
        output["direction_cutoff"] = cutoff
        output["confidence_polarity"] = "HIGH_MARGIN"
        output["confidence_threshold"] = threshold
        output["confirmation_block"] = block
        return output

    @staticmethod
    def complete_block(
        state: dict[str, Any], scored: pd.DataFrame, outcomes: pd.DataFrame,
    ) -> dict[str, Any]:
        assert len(scored) == evaluation.BLOCK_SIZE
        assert set(outcomes.columns) == {"event_id", "y", "fwd_ret_30m"}
        updated = copy.deepcopy(state)
        machine = updated["machine_state"]
        block = int(machine["next_confirmation_block"])
        machine["completed_confirmation_blocks"].append({"block": block, "n": len(scored)})
        machine["confirmation_rows_completed"] += len(scored)
        machine["next_confirmation_block"] += 1
        machine["last_completed_ids"] = scored.event_id.astype(str).tolist()
        machine["last_completed_y"] = outcomes.set_index("event_id").loc[scored.event_id, "y"].astype(int).tolist()
        return updated


def prepare_workspace(tmp_path: Path) -> tuple[Path, evaluation.FrozenRuntime]:
    policy_target = tmp_path / evaluation.POLICY_REL
    policy_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ROOT / evaluation.POLICY_REL, policy_target)
    frozen_target = tmp_path / evaluation.FROZEN_REL
    frozen_target.mkdir(parents=True, exist_ok=True)
    for name in (
        evaluation.FROZEN_SHA_NAME,
        evaluation.FROZEN_SCORER_NAME,
        evaluation.FROZEN_MODEL_NAME,
        evaluation.FROZEN_SPEC_NAME,
        evaluation.FROZEN_STATE_NAME,
    ):
        shutil.copy2(SOURCE_ROOT / evaluation.FROZEN_REL / name, frozen_target / name)
    manifest = json.loads((frozen_target / evaluation.FROZEN_SHA_NAME).read_text(encoding="utf-8"))
    initial_state = {
        "machine_state": {
            "completed_confirmation_blocks": [],
            "confirmation_rows_completed": 0,
            "next_confirmation_block": 1,
        }
    }
    scorer = FakeFrozenScorer()
    runtime = evaluation.FrozenRuntime(
        scorer=scorer,
        bundle=None,
        initial_state=initial_state,
        spec={
            "deterministic_self_test": {
                "state_fingerprint_before": scorer.state_fingerprint(initial_state),
            }
        },
        freeze_manifest=manifest,
        freeze_manifest_sha256=evaluation.PINNED_FROZEN_SHA256,
    )
    return tmp_path, runtime


def block_frame(block: int) -> pd.DataFrame:
    start_order = (block - 1) * evaluation.BLOCK_SIZE + 1
    start_time = pd.Timestamp("2026-09-01T14:00:00Z") + pd.Timedelta(days=block - 1)
    rows = []
    for offset in range(evaluation.BLOCK_SIZE):
        order = start_order + offset
        rows.append({
            "prospective_observation_order": order,
            "prospective_role_assignment_order": order,
            "event_id": f"SEC:TEST:{order:04d}",
            "market": "US",
            "ticker": "ABCD",
            "event_time_utc": (start_time + pd.Timedelta(minutes=offset)).isoformat(),
            "source": "SEC_V224_PROSPECTIVE",
            "source_family": "US_SEC",
            "form": "8-K",
            "event_type": "SEC_8-K",
            "headline": f"Test filing {order}",
            "body": "Item 8.01 Other Events",
            "pre_ret_1": (offset - 10) / 1000,
            "y": offset % 2,
            "fwd_ret_30m": 0.01 if offset % 2 else -0.005,
            "entry_slippage_sec": 0,
            "exit_slippage_sec": 0,
            "actual_hold_seconds": 1800,
            "data_role": "V224_PROSPECTIVE_CONFIRMATION",
        })
    return pd.DataFrame(rows)


def write_checkpoints(root: Path, blocks: int) -> None:
    cumulative: list[pd.DataFrame] = []
    for block in range(1, blocks + 1):
        cumulative_n = block * evaluation.BLOCK_SIZE
        frame = block_frame(block)
        manifest_path, snapshot_path = evaluation._checkpoint_paths(root, cumulative_n)
        evaluation._atomic_replace(snapshot_path, evaluation.deterministic_csv_gzip_bytes(frame))
        reloaded = pd.read_csv(snapshot_path, low_memory=False)
        cumulative.append(reloaded)
        manifest = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CONFIRMATION",
            "checkpoint_rows": cumulative_n,
            "rows": evaluation.BLOCK_SIZE,
            "cumulative_rows": cumulative_n,
            "block_start_observation_order": cumulative_n - evaluation.BLOCK_SIZE + 1,
            "block_end_observation_order": cumulative_n,
            "policy_sha256": evaluation.PINNED_POLICY_SHA256,
            "prefix_records_sha256": evaluation.canonical_records_sha256(pd.concat(cumulative, ignore_index=True, sort=False)),
            "block_records_sha256": evaluation.canonical_records_sha256(reloaded),
            "snapshot_path": str(snapshot_path.relative_to(root)).replace("\\", "/"),
            "snapshot_sha256": evaluation.sha256_file(snapshot_path),
            "source_family": "US_SEC",
            "contract_version": "EXACT_T2_30M_V36",
            "protocol_change_authority": False,
            "seal_or_final_open_authority": False,
            "v224_evaluation_run_by_watcher": False,
        }
        evaluation._atomic_replace(manifest_path, evaluation.pretty_json_bytes(manifest))


def tree_hashes(root: Path) -> dict[str, str]:
    directory = root / evaluation.OUTPUT_REL
    return {
        str(path.relative_to(directory)): evaluation.sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_zero_checkpoint_status_is_byte_idempotent(tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    first = evaluation.run(root, runtime=runtime)
    status_path = root / evaluation.OUTPUT_REL / "EVALUATION_STATUS.json"
    before = status_path.read_bytes()
    second = evaluation.run(root, runtime=runtime)

    assert first == second
    assert first["status"] == "ZERO_EXACT_ROWS_NO_CHECKPOINT_READY"
    assert first["prospective_outcome_rows_read"] == 0
    assert first["research_seal_rows_read"] == 0
    assert first["final_meta_rows_read"] == 0
    assert first["central_authority_rows_read"] == 0
    assert status_path.read_bytes() == before
    assert not (root / evaluation.OUTPUT_REL / "blocks").exists()


def test_two_blocks_are_prequential_append_only_and_fully_idempotent(tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    write_checkpoints(root, blocks=2)
    first = evaluation.run(root, runtime=runtime)
    before = tree_hashes(root)
    second = evaluation.run(root, runtime=runtime)
    after = tree_hashes(root)

    assert first == second
    assert before == after
    assert first["completed_checkpoint_rows"] == [20, 40]
    assert first["prediction_rows"] == 40
    assert first["prospective_outcome_rows_read"] == 40
    prediction_1 = pd.read_csv(root / evaluation.OUTPUT_REL / "blocks/BLOCK_001_PREDICTIONS.csv.gz")
    prediction_2 = pd.read_csv(root / evaluation.OUTPUT_REL / "blocks/BLOCK_002_PREDICTIONS.csv.gz")
    assert not set(prediction_1.columns).intersection(evaluation.OUTCOME_COLUMNS)
    assert prediction_1.direction_cutoff.unique().tolist() == [0.50]
    assert prediction_2.direction_cutoff.unique().tolist() == [0.51]
    state_1 = json.loads((root / evaluation.OUTPUT_REL / "blocks/STATE_AFTER_BLOCK_001.json").read_text())
    state_2 = json.loads((root / evaluation.OUTPUT_REL / "blocks/STATE_AFTER_BLOCK_002.json").read_text())
    assert state_1["machine_state"]["confirmation_rows_completed"] == 20
    assert state_2["machine_state"]["confirmation_rows_completed"] == 40
    assert (root / evaluation.OUTPUT_REL / "N020_METRICS.json").exists()
    assert (root / evaluation.OUTPUT_REL / "N040_REPORT.md").exists()
    assert not (root / evaluation.OUTPUT_REL / "N060_STATUS.json").exists()
    completion = json.loads((root / evaluation.OUTPUT_REL / "blocks/BLOCK_002_COMPLETION.json").read_text())
    assert completion["outcomes_opened_after_prediction_persist"] is True
    assert completion["state_append_only"] is True


def test_checkpoint_snapshot_hash_tamper_fails_before_prediction(tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    write_checkpoints(root, blocks=1)
    _, snapshot = evaluation._checkpoint_paths(root, 20)
    snapshot.write_bytes(snapshot.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="WATCHER_CHECKPOINT_MANIFEST_GUARD_FAIL"):
        evaluation.run(root, runtime=runtime)
    assert not (root / evaluation.OUTPUT_REL / "blocks/BLOCK_001_PREDICTIONS.csv.gz").exists()


def test_frozen_sha_tamper_fails_closed(tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    frozen_state = root / evaluation.FROZEN_REL / evaluation.FROZEN_STATE_NAME
    frozen_state.write_bytes(frozen_state.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="V224_FROZEN_FILE_GUARD_FAIL"):
        evaluation.run(root, runtime=runtime)


def test_n60_report_is_non_decisional_and_only_n100_report_is_final(tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    rows = pd.DataFrame({
        "event_id": [f"SEC:METRIC:{index:03d}" for index in range(100)],
        "event_time_utc": [
            (pd.Timestamp("2026-09-01T14:00:00Z") + pd.Timedelta(days=index // 20, minutes=index % 20)).isoformat()
            for index in range(100)
        ],
        "confirmation_block": [index // 20 + 1 for index in range(100)],
        "y": [index % 2 for index in range(100)],
        "fwd_ret_30m": [0.01 if index % 2 else -0.005 for index in range(100)],
        "prediction": [bool(index % 2) for index in range(100)],
        "high_conf": [index % 5 == 0 for index in range(100)],
        "probability": [0.7 if index % 2 else 0.3 for index in range(100)],
        "confidence_signal": [0.2 if index % 5 == 0 else 0.1 for index in range(100)],
    })
    evaluation.write_checkpoint_reports(root, 60, rows.iloc[:60], runtime.initial_state, runtime)
    evaluation.write_checkpoint_reports(root, 100, rows, runtime.initial_state, runtime)
    sixty = evaluation.checkpoint_classification(60)
    hundred = evaluation.checkpoint_classification(100)
    sixty_status = json.loads((root / evaluation.OUTPUT_REL / "N060_STATUS.json").read_text())
    hundred_status = json.loads((root / evaluation.OUTPUT_REL / "N100_STATUS.json").read_text())
    hundred_metrics = json.loads((root / evaluation.OUTPUT_REL / "N100_METRICS.json").read_text())
    assert sixty["classification"] == "PRELIMINARY_60_NON_DECISIONAL"
    assert sixty["decision_authority"] == "NONE"
    assert sixty["final_prospective_checkpoint"] is False
    assert sixty_status["decision_authority"] == "NONE"
    assert hundred["classification"] == "FINAL_PROSPECTIVE_CONFIRMATION_REPORT"
    assert hundred["final_prospective_checkpoint"] is True
    assert hundred["decision_authority"] == "PROSPECTIVE_CONFIRMATION_ONLY"
    assert hundred_status["final_prospective_checkpoint"] is True
    assert "only N=100" in (root / evaluation.OUTPUT_REL / "N100_REPORT.md").read_text(encoding="utf-8")
    assert hundred_metrics["aggregate"]["bootstrap"]["n_boot"] == 2000
    assert hundred_metrics["aggregate"]["bootstrap"]["seed"] == 20260830
    assert hundred_metrics["aggregate"]["bootstrap"]["resampling_unit"] == "EVENT_DATE"
    assert hundred_metrics["aggregate"]["bootstrap"]["balanced_accuracy_lower95"] == 1.0
    assert hundred_metrics["aggregate"]["bootstrap"]["highconf_net_lower95"] > 0
    monotonicity = hundred_metrics["aggregate"]["confidence_bin_monotonicity"]
    assert len(monotonicity["bins"]) == 5
    assert monotonicity["nondecreasing"] is True
    markdown = (root / evaluation.OUTPUT_REL / "N100_REPORT.md").read_text(encoding="utf-8")
    assert "bootstrap_balanced_accuracy_lower95" in markdown
    assert "bootstrap_highconf_net_lower95" in markdown
    assert "confidence_bin_nondecreasing_fraction" in markdown
    assert "—" not in markdown


def test_single_class_auc_ba_and_empty_highconf_bootstrap_are_safe() -> None:
    rows = pd.DataFrame({
        "event_time_utc": pd.date_range("2026-09-01T14:00:00Z", periods=20, freq="1min"),
        "confirmation_block": [1] * 20,
        "y": [0] * 20,
        "fwd_ret_30m": [-0.005] * 20,
        "prediction": [False] * 20,
        "high_conf": [False] * 20,
        "probability": [0.3] * 20,
        "confidence_signal": [0.2] * 20,
    })
    first = evaluation.metrics_for(rows, 0.002, include_bootstrap=True)
    second = evaluation.metrics_for(rows, 0.002, include_bootstrap=True)
    assert first == second
    assert first["auc"] is None
    assert first["balanced_accuracy"] == 1.0
    assert first["confidence_correctness_auc"] is None
    assert first["highconf_net"] is None
    assert first["bootstrap"]["balanced_accuracy_lower95"] == 1.0
    assert first["bootstrap"]["highconf_net_lower95"] is None
    assert first["bootstrap"]["highconf_net_valid_replicates"] == 0
