from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import v224_prospective_causal_one_shot as track_a
import v224_prospective_locked_evaluation as locked
import v224_prospective_price_collect as price_collect


SOURCE_ROOT = Path(__file__).resolve().parent


class FakeScorer:
    NUMERIC_COLS = ["pre_ret_1"]
    FORBIDDEN_SCORE_COLUMNS = {"y", "fwd_ret_30m", "exit_price"}
    COST = 0.002

    @staticmethod
    def canonical_json_bytes(value: Any) -> bytes:
        return locked.canonical_json_bytes(value)

    @staticmethod
    def state_fingerprint(state: dict[str, Any]) -> str:
        return locked.sha256_bytes(locked.canonical_json_bytes(state))

    @staticmethod
    def score_digest(frame: pd.DataFrame) -> str:
        columns = [
            "event_id", "raw_probability", "probability", "prediction", "margin",
            "high_conf", "confidence_signal", "polarity_reversed", "direction_cutoff",
            "confidence_polarity", "confidence_threshold", "confirmation_block",
        ]
        records = track_a.clean_json(frame[columns].to_dict(orient="records"))
        return locked.sha256_bytes(locked.canonical_json_bytes(records))

    @staticmethod
    def score_block(features: pd.DataFrame, bundle: Any, state: dict[str, Any]) -> pd.DataFrame:
        assert not set(features.columns).intersection(FakeScorer.FORBIDDEN_SCORE_COLUMNS)
        ordered = features.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
        block = int(state["machine_state"]["next_confirmation_block"])
        raw = np.linspace(0.40, 0.60, len(ordered))
        reverse = block % 2 == 1
        oriented = 1 - raw if reverse else raw
        cutoff = 0.51
        final = np.clip(0.5 + oriented - cutoff, 0, 1)
        margin = np.abs(final - 0.5)
        output = ordered[["event_id", "event_time_utc", "source_family", "form", "event_type"]].copy()
        output["numeric_probability"] = raw
        output["text_probability"] = raw
        output["semantic_structure_probability"] = raw
        output["raw_probability"] = raw
        output["oriented_probability"] = oriented
        output["probability"] = final
        output["prediction"] = final >= 0.5
        output["margin"] = margin
        output["high_conf"] = margin >= float(np.quantile(margin, 0.8))
        output["confidence_signal"] = margin
        output["polarity_reversed"] = reverse
        output["direction_cutoff"] = cutoff
        output["confidence_polarity"] = "HIGH_MARGIN"
        output["confidence_threshold"] = float(np.quantile(margin, 0.8))
        output["confirmation_block"] = block
        return output

    @staticmethod
    def complete_block(state: dict[str, Any], scored: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
        updated = copy.deepcopy(state)
        machine = updated["machine_state"]
        block = int(machine["next_confirmation_block"])
        machine["completed_confirmation_blocks"].append({"block": block, "n": len(scored)})
        machine["confirmation_rows_completed"] += len(scored)
        machine["next_confirmation_block"] += 1
        machine["previous_completed_ids"] = scored.event_id.astype(str).tolist()
        machine["previous_completed_y"] = outcomes.y.astype(int).tolist()
        return updated


def prepare_workspace(tmp_path: Path) -> tuple[Path, locked.FrozenRuntime]:
    policy_target = tmp_path / locked.POLICY_REL
    policy_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ROOT / locked.POLICY_REL, policy_target)
    frozen_target = tmp_path / locked.FROZEN_REL
    frozen_target.mkdir(parents=True, exist_ok=True)
    for name in (
        locked.FROZEN_SHA_NAME, locked.FROZEN_SCORER_NAME, locked.FROZEN_MODEL_NAME,
        locked.FROZEN_SPEC_NAME, locked.FROZEN_STATE_NAME,
    ):
        shutil.copy2(SOURCE_ROOT / locked.FROZEN_REL / name, frozen_target / name)
    manifest = json.loads((frozen_target / locked.FROZEN_SHA_NAME).read_text(encoding="utf-8"))
    state = {
        "machine_state": {
            "completed_confirmation_blocks": [],
            "confirmation_rows_completed": 0,
            "next_confirmation_block": 1,
        }
    }
    scorer = FakeScorer()
    runtime = locked.FrozenRuntime(
        scorer=scorer,
        bundle=None,
        initial_state=state,
        spec={"deterministic_self_test": {"state_fingerprint_before": scorer.state_fingerprint(state)}},
        freeze_manifest=manifest,
        freeze_manifest_sha256=locked.PINNED_FROZEN_SHA256,
    )
    return tmp_path, runtime


def write_role_ledger(root: Path, count: int) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-09-01T14:30:00Z")
    for index in range(count):
        rows.append({
            "assignment_order": index + 1,
            "event_id": f"SEC:CAUSAL:{index + 1:04d}",
            "market": "US",
            "ticker": "ABCD",
            "company": "Causal Biopharma Inc.",
            "event_time_utc": (base + pd.Timedelta(minutes=index)).isoformat(),
            "source": "SEC_V224_PROSPECTIVE",
            "source_family": "US_SEC",
            "form": "8-K",
            "headline": f"Item 8.01 event {index + 1}",
            "body": "Item 8.01 Other Events. Phase 2 endpoint met.",
            "event_type": "SEC_8-K",
            "timestamp_quality": "EXACT",
            "url": "https://www.sec.gov/test",
            "data_role": track_a.ROLE,
            "role_namespace": "INDEPENDENT_APPEND_ONLY_V224",
            "role_frozen_at_utc": "2026-08-30T00:00:00Z",
            "role_policy_sha256": locked.PINNED_POLICY_SHA256,
            "raw_event_file": "data/test.csv",
            "raw_event_file_sha256": "0" * 64,
        })
    columns = [
        "assignment_order", "event_id", "market", "ticker", "company",
        "event_time_utc", "source", "source_family", "form", "headline", "body",
        "event_type", "timestamp_quality", "url", "data_role", "role_namespace",
        "role_frozen_at_utc", "role_policy_sha256", "raw_event_file",
        "raw_event_file_sha256",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    path = root / track_a.ROLE_LEDGER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    locked._atomic_replace(path, locked.deterministic_csv_gzip_bytes(frame))
    return frame


def write_price_day(root: Path, through: str = "2026-09-01T20:00:00Z") -> Path:
    timestamps = pd.date_range("2026-09-01T13:30:00Z", through, freq="1min")
    opens = 10 + np.arange(len(timestamps)) * 0.001
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": opens + 0.01,
        "low": opens - 0.01,
        "close": opens + 0.002,
        "volume": np.full(len(timestamps), 1000),
    })
    path = root / "price_data/V224_PROSPECTIVE_MASSIVE_1M/2026/ABCD/2026-09-01.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def tree_hashes(root: Path) -> dict[str, str]:
    directory = root / track_a.TRACK_REL
    return {
        str(path.relative_to(directory)): locked.sha256_file(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def test_zero_rows_bootstrap_is_one_shot_and_byte_idempotent(tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    write_role_ledger(root, 0)
    first = track_a.run(root, runtime)
    before = tree_hashes(root)
    second = track_a.run(root, runtime)
    assert first == second
    assert before == tree_hashes(root)
    assert first["status"] == "ZERO_PROSPECTIVE_ROWS_BOOTSTRAPPED"
    assert first["old_outcome_first_watcher_used"] is False
    assert first["legacy_locked_evaluator_entrypoint_run_used"] is False
    assert first["shared_frozen_verification_metric_helpers_imported"] is True
    assert first["central_authority_rows_read"] == 0


def test_causal_feature_projection_never_loads_entry_future_fields_or_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, _ = prepare_workspace(tmp_path)
    roles = write_role_ledger(root, 1)
    price = write_price_day(root)
    calls: list[tuple[list[str], list[tuple[str, str, Any]]]] = []
    original = track_a.projected_parquet

    def observed(path: Path, columns: list[str], filters: list[tuple[str, str, Any]]) -> pd.DataFrame:
        calls.append((list(columns), list(filters)))
        return original(path, columns, filters)

    monkeypatch.setattr(track_a, "projected_parquet", observed)
    feature, reason, guards = track_a.causal_feature_snapshot(roles.iloc[0], price)
    assert reason == "OK"
    assert feature is not None
    assert not track_a.OUTCOME_NAMES.intersection(feature)
    assert calls[0][0] == ["timestamp", "open"]
    assert calls[1][0] == ["timestamp", "open", "high", "low", "close", "volume"]
    assert calls[1][1][1][1:] == ("<", pd.Timestamp(feature["actual_entry_time_utc"]))
    assert guards["future_bar_rows_loaded"] == 0
    assert guards["entry_candle_future_fields_loaded"] is False
    assert pd.Timestamp(guards["latest_prehistory_bar_utc"]) < pd.Timestamp(feature["actual_entry_time_utc"])


def test_full_block_predictions_are_durable_before_any_outcome_and_rerun_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    write_role_ledger(root, 20)
    write_price_day(root)
    original = track_a.open_exact_outcome

    def guarded_open(feature: dict[str, Any], price_path: Path):
        prediction = track_a.block_paths(root, 1)["predictions"]
        manifest = track_a.block_paths(root, 1)["prediction_manifest"]
        assert prediction.is_file() and manifest.is_file()
        assert all(track_a.event_prediction_audit_path(root, order).is_file() for order in range(1, 21))
        return original(feature, price_path)

    monkeypatch.setattr(track_a, "open_exact_outcome", guarded_open)
    first = track_a.run(root, runtime)
    before = tree_hashes(root)
    second = track_a.run(root, runtime)
    after = tree_hashes(root)

    assert first == second
    assert before == after
    assert first["completed_rows"] == 20
    assert first["prediction_rows"] == 20
    assert first["outcome_rows_opened"] == 20
    predictions = pd.read_csv(track_a.block_paths(root, 1)["predictions"])
    assert not set(predictions.columns).intersection(track_a.OUTCOME_NAMES)
    for order in range(1, 21):
        prediction_audit = json.loads(track_a.event_prediction_audit_path(root, order).read_text())
        outcome_audit = json.loads(track_a.event_outcome_path(root, order).read_text())
        assert pd.Timestamp(prediction_audit["prediction_created_at_utc"]) <= pd.Timestamp(outcome_audit["outcome_opened_at_utc"])
        assert outcome_audit["outcome_opened_after_full_block_prediction_durable"] is True
        for field in (
            "feature_row_sha256", "frozen_scorer_sha256", "block_prediction_file_sha256",
            "event_prediction_sha256", "raw_probability", "final_probability",
            "raw_direction", "final_direction", "polarity_reversed",
            "confidence_polarity", "confidence_signal",
        ):
            assert field in prediction_audit
    metrics_path = root / track_a.TRACK_REL / "N020_RAW_VS_ADAPTED_METRICS.json"
    metrics = json.loads(metrics_path.read_text())
    assert set(metrics["comparison"]) == {"raw", "adapted", "adapted_minus_raw"}
    assert metrics["prediction_before_outcomes"] is True
    assert track_a.block_paths(root, 1)["state"].exists()


def test_missing_exit_bars_leave_durable_predictions_and_fail_closed_block_state(tmp_path: Path) -> None:
    root, runtime = prepare_workspace(tmp_path)
    write_role_ledger(root, 20)
    # Enough for all entries (last event entry 14:51), but no +30m exits.
    write_price_day(root, through="2026-09-01T14:55:00Z")
    status = track_a.run(root, runtime)
    assert status["status"] == "PREDICTIONS_DURABLE_AWAITING_EXACT_OUTCOMES"
    assert status["prediction_rows"] == 20
    assert status["completed_rows"] == 0
    assert status["blocker"] == "BLOCK_001_PREDICTIONS_DURABLE_AWAITING_ALL_EXACT_OUTCOMES"
    assert track_a.block_paths(root, 1)["predictions"].exists()
    assert not track_a.block_paths(root, 1)["state"].exists()
    assert not track_a.block_paths(root, 1)["completion"].exists()
    assert status["outcome_reason_counts"]["EXIT_BAR_MISSING"] > 0


def test_exact_outcome_timing_guard_rejects_more_than_60_second_slippage(tmp_path: Path) -> None:
    root, _ = prepare_workspace(tmp_path)
    roles = write_role_ledger(root, 1)
    price = write_price_day(root)
    feature, reason, _ = track_a.causal_feature_snapshot(roles.iloc[0], price)
    assert reason == "OK" and feature is not None
    frame = pd.read_parquet(price)
    exit_target = pd.Timestamp(feature["actual_entry_time_utc"]) + pd.Timedelta(minutes=30)
    frame = frame[~frame.timestamp.isin([exit_target, exit_target + pd.Timedelta(minutes=1)])]
    frame.to_parquet(price, index=False)
    artifact = {"feature_row": feature}
    outcome, outcome_reason = track_a.open_exact_outcome(artifact, price)
    assert outcome is None
    assert outcome_reason == "EXIT_BAR_MISSING"


def test_price_collector_zero_roles_uses_no_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, _ = prepare_workspace(tmp_path)
    write_role_ledger(root, 0)
    monkeypatch.setenv("MASSIVE_API_KEY", "test-secret-not-logged")

    def forbidden_fetch(*args):
        raise AssertionError("zero-role collector must not call provider")

    first = price_collect.run(root, max_requests=200, fetcher=forbidden_fetch)
    status_path = root / price_collect.STATUS_REL
    before = status_path.read_bytes()
    second = price_collect.run(root, max_requests=200, fetcher=forbidden_fetch)
    assert first == second
    assert status_path.read_bytes() == before
    assert first["status"] == "ZERO_ROLE_ROWS_NO_REQUESTS"
    assert first["requests_used"] == 0
    assert first["labels_or_returns_loaded"] is False


def test_price_collector_is_bounded_and_resume_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, _ = prepare_workspace(tmp_path)
    roles = write_role_ledger(root, 2)
    roles.loc[1, "ticker"] = "EFGH"
    locked._atomic_replace(root / track_a.ROLE_LEDGER_REL, locked.deterministic_csv_gzip_bytes(roles))
    monkeypatch.setenv("MASSIVE_API_KEY", "test-secret-not-logged")
    timestamps = pd.date_range("2026-09-01T13:30:00Z", "2026-09-01T20:00:00Z", freq="1min")
    opens = 10 + np.arange(len(timestamps)) * 0.001
    bars = pd.DataFrame({
        "timestamp": timestamps, "open": opens, "high": opens + 0.01,
        "low": opens - 0.01, "close": opens + 0.002,
        "volume": np.full(len(timestamps), 1000),
    })
    calls: list[tuple[str, str]] = []

    def fake_fetch(ticker: str, trade_date: str, key: str):
        assert key == "test-secret-not-logged"
        calls.append((ticker, trade_date))
        return bars.copy(), {"ok": True}

    first = price_collect.run(root, max_requests=1, fetcher=fake_fetch)
    assert first["requests_used"] == 1
    assert first["raw_ticker_days_fetched"] == 1
    assert first["deferred"] == 1
    assert len(calls) == 1
    second = price_collect.run(root, max_requests=2, fetcher=fake_fetch)
    assert second["requests_used"] == 1
    assert second["cached_complete"] == 1
    assert second["raw_ticker_days_fetched"] == 1
    assert len(calls) == 2
    before = tree_hashes(root)
    third = price_collect.run(root, max_requests=2, fetcher=fake_fetch)
    assert third["requests_used"] == 0
    assert third["cached_complete"] == 2
    assert len(calls) == 2
    assert third["labels_or_returns_loaded"] is False
    assert third["outcomes_computed"] is False
    assert third["central_authority_rows_read"] == 0
