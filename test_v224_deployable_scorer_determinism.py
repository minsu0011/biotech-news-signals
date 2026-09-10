"""Determinism and label-blindness checks for the frozen V224 scorer."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import experiment_v223_exact_sec_semantic_transfer as v223
import freeze_v224_deployable_scorer as freeze
import v224_deployable_scorer as scorer


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "cert_research" / "V224_CONFIRMATION"


def _published_inputs() -> tuple[dict, dict, dict, pd.DataFrame]:
    spec = json.loads((FROZEN / freeze.SPEC_NAME).read_text(encoding="utf-8"))
    state = json.loads((FROZEN / freeze.STATE_NAME).read_text(encoding="utf-8"))
    bundle = scorer.load_bundle(FROZEN / freeze.MODEL_NAME)
    _, extension, _, _ = v223.load_data()
    wanted = spec["deterministic_self_test"]["event_ids"]
    indexed = extension.set_index(extension.event_id.astype(str), drop=False)
    sample = indexed.loc[wanted].reset_index(drop=True)
    return spec, state, bundle, sample


def test_published_hashes_full_pipeline_and_state_update_are_deterministic() -> None:
    manifest = json.loads((FROZEN / freeze.SHA_NAME).read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = FROZEN / name
        assert path.stat().st_size == record["bytes"]
        assert freeze.sha256_file(path) == record["sha256"]
    assert (FROZEN / freeze.SCORER_NAME).read_bytes() == (ROOT / "v224_deployable_scorer.py").read_bytes()

    spec, state, bundle, sample = _published_inputs()
    label_blind = freeze.feature_only(sample)
    before = scorer.state_fingerprint(state)
    first = scorer.score_block(label_blind, bundle, state)
    second = scorer.score_block(label_blind.sample(frac=1.0, random_state=224), bundle, state)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert scorer.state_fingerprint(state) == before
    assert scorer.score_digest(first) == spec["deterministic_self_test"]["scored_output_sha256"]
    assert before == spec["deterministic_self_test"]["state_fingerprint_before"]

    outcomes = sample[["event_id", "y", "fwd_ret_30m"]].copy()
    next_a = scorer.complete_block(copy.deepcopy(state), first, outcomes)
    next_b = scorer.complete_block(copy.deepcopy(state), second, outcomes.sample(frac=1.0, random_state=224))
    assert scorer.canonical_json_bytes(next_a) == scorer.canonical_json_bytes(next_b)
    assert scorer.derive_policy(next_a) == scorer.derive_policy(next_b)
    assert next_a["machine_state"]["confirmation_rows_completed"] == 20
    assert next_a["machine_state"]["next_confirmation_block"] == 2


def test_score_time_labels_and_non_sec_are_rejected() -> None:
    _, state, bundle, sample = _published_inputs()
    with pytest.raises(ValueError, match="LABEL_OR_OUTCOME_COLUMNS_FORBIDDEN"):
        scorer.score_block(sample, bundle, state)
    non_sec = freeze.feature_only(sample)
    non_sec.loc[0, "source_family"] = "US_NEWS"
    with pytest.raises(ValueError, match="REJECTS_NON_US_SEC"):
        scorer.score_block(non_sec, bundle, state)


def test_cross_process_frozen_inference_and_hash_verification() -> None:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = freeze.REQUIRED_PYTHONHASHSEED
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "freeze_v224_deployable_scorer.py"), "--verify-existing"],
        cwd=ROOT, env=environment, check=False, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["deterministic_inference_matches"] is True
    assert result["artifact_hashes_match"] is True
    assert result["source_dependency_hashes_match"] is True
    assert result["output_written"] is False
