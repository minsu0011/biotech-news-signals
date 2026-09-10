"""Freeze the complete V224 scorer and its initial prospective DEV state.

Only opened DEV/V223/V224 research artifacts are read.  The script never
reads Research Seal or Final Meta and never changes the frozen V224 algorithm.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn

import experiment_v223_exact_sec_semantic_transfer as v223
import experiment_v224_causal_polarity_confidence_guard as v224
import experiment_v36_robust as v36
import experiment_v38_opportunity as v38
import experiment_v59_sec_structured_v1 as v59
import runtime_limits
import v224_deployable_scorer as scorer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "cert_research" / "V224_CONFIRMATION"
SCORER_SOURCE = ROOT / "v224_deployable_scorer.py"
TEST_SOURCE = ROOT / "test_v224_deployable_scorer_determinism.py"
MODEL_NAME = "FROZEN_V224_DEPLOYABLE_SCORER.pkl.gz"
SCORER_NAME = "FROZEN_V224_DEPLOYABLE_SCORER.py"
SPEC_NAME = "FROZEN_V224_SPEC.json"
STATE_NAME = "FROZEN_V224_STATE.json"
SHA_NAME = "FROZEN_V224_SHA256.json"

PINNED = {
    "experiment_v223_exact_sec_semantic_transfer.py": "730e7d03cd0500fecf4a285332c52de516f3b426d53681f8cd28a46987c3d2f3",
    "experiment_v224_causal_polarity_confidence_guard.py": "d1da5f0e62c89840cc33e6ecb0488114cfe336bf58b32c092fbb9265fd598df1",
    "output_V223/ARTIFACT_MANIFEST.json": "86c1b47042fbdf2dec9443df99b5b6fd3552405a125054bd1970676d4bdf737b",
    "output_V223/COMMIT.json": "3cd1128fb1bc0893f4f8404b935f730f3a9836c3cdae817a387e41de26cda7bc",
    "output_V223/V223_EXACT_SEC_SEMANTIC_TRANSFER_REPORT.json": "783c1cfed484308b0edcacdd2f40f325d91ccbae5a05a225103b2de31182b79d",
    "output_V223/V223_FROZEN_V59_N_T_S_DIAGNOSTIC_PREDICTIONS.csv.gz": "85d3382bce4e2469d506e62e7e59feca011ba9d84a025e3a982dedc18d57e916",
    "output_V224/ARTIFACT_MANIFEST.json": "90f96de25433610d447d78a1e993a76b91418096de457f638dd67cdbd96c9d4e",
    "output_V224/COMMIT.json": "3261d5ec77ef6ff20acc22f345cafbd7034e06ec933bf4d55e37df8244af0592",
    "output_V224/V224_CAUSAL_POLARITY_CONFIDENCE_GUARD_REPORT.json": "97de3680880438c936be0deed2fe43b153e7d3af28c53eeba848ede0dfb5aff6",
    "output_V224/V224_EXACT_SEC_N_T_S_GUARDED_PREDICTIONS.csv.gz": "addd3b19f85f95126eb6fe0da538db94d95d2a1f659fd3f8d168535b344c7b43",
    "data/dev_contract_v36_labeled.csv.gz": "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    "data/V59_DEV_EXTENSION_LABELED.csv.gz": "e11c60ef38c5b65e676bab79c96cd21e44fdcb6d604e02a550ffc49003ed311b",
    "research/NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json": "e3addcd52e36bb6316c1947df6acfdf60460b304df48c7ab8660fb3f012535d2",
    "research/V221_US_SEC_EXACT_ACCESSION_TEXT_MANIFEST.csv": "96b2e108dcf673071b6520e261a96b832e4f5314c2397974a542bb108c939d89",
    "output_V59/MODEL_COMPARISON.json": "5a572e92f8bf8e0d6b6bc7d017cca20f6fda50a28054204615e5e86383d483fe",
}
REQUIRED_PYTHONHASHSEED = "0"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False, default=str) + "\n").encode("utf-8")


def canonical_digest(value: Any) -> str:
    return sha256_bytes(scorer.canonical_json_bytes(value))


def verify_pins() -> None:
    for relative, expected in PINNED.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"V224_FREEZE_PIN_MISMATCH:{relative}:{actual}")
    for version in (223, 224):
        out = ROOT / f"output_V{version}"
        manifest_path = out / "ARTIFACT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, record in manifest["files"].items():
            path = out / name
            if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"V224_FREEZE_COMMITTED_FILE_MISMATCH:V{version}:{name}")
        commit = json.loads((out / "COMMIT.json").read_text(encoding="utf-8"))
        if commit["manifest_sha256"] != sha256_file(manifest_path):
            raise RuntimeError(f"V224_FREEZE_MANIFEST_COMMIT_MISMATCH:V{version}")
    v223_commit = json.loads((ROOT / "output_V223" / "COMMIT.json").read_text(encoding="utf-8"))
    v224_commit = json.loads((ROOT / "output_V224" / "COMMIT.json").read_text(encoding="utf-8"))
    if v223_commit["runner_sha256"] != PINNED["experiment_v223_exact_sec_semantic_transfer.py"]:
        raise RuntimeError("V224_FREEZE_V223_RUNNER_COMMIT_MISMATCH")
    if v224_commit["runner_sha256"] != PINNED["experiment_v224_causal_polarity_confidence_guard.py"]:
        raise RuntimeError("V224_FREEZE_V224_RUNNER_COMMIT_MISMATCH")
    if v224_commit["v223_commit_sha256"] != PINNED["output_V223/COMMIT.json"]:
        raise RuntimeError("V224_FREEZE_CHAIN_OF_CUSTODY_MISMATCH")
    report = json.loads((ROOT / "output_V224" / "V224_CAUSAL_POLARITY_CONFIDENCE_GUARD_REPORT.json").read_text(encoding="utf-8"))
    if report["primary_candidate"] != scorer.MODEL_FAMILY:
        raise RuntimeError("V224_FREEZE_PRIMARY_CANDIDATE_MISMATCH")
    if report["status"] != "DEV_RESEARCH_SURVIVOR_PENDING_INDEPENDENT_CONFIRMATION":
        raise RuntimeError("V224_FREEZE_STATUS_MISMATCH")
    guards = report["guards"]
    if guards["research_seal_or_final_rows_read"] or guards["seal_authorized"] or guards["final_meta_authorized"]:
        raise RuntimeError("V224_FREEZE_RESEARCH_BOUNDARY_FAIL")


def _assert_feature_equivalence(training: pd.DataFrame, training_sec: pd.DataFrame) -> pd.DataFrame:
    original_numeric = v36.feature_frame(training)[scorer.NUMERIC_COLS + list(scorer.CAT_COLUMNS)]
    frozen_numeric = scorer.numeric_feature_frame(training)
    pd.testing.assert_frame_equal(original_numeric.reset_index(drop=True), frozen_numeric,
                                  check_dtype=False, check_exact=True)
    prepared_sec = scorer.prepare_sec_features(training_sec)
    if not prepared_sec.sec_segment_text.equals(training_sec.sec_segment_text.reset_index(drop=True)):
        raise RuntimeError("V224_FREEZE_SEC_SEGMENT_PREPROCESSING_DRIFT")
    original_structures = [json.dumps(value, sort_keys=True, allow_nan=False)
                           for value in training_sec.sec_structure.tolist()]
    frozen_structures = [json.dumps(value, sort_keys=True, allow_nan=False)
                         for value in prepared_sec.sec_structure.tolist()]
    if original_structures != frozen_structures:
        raise RuntimeError("V224_FREEZE_STRUCTURED_SEMANTIC_PREPROCESSING_DRIFT")
    return prepared_sec


def fit_bundle() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    legacy, extension, epoch, _ = v223.load_data()
    training = pd.concat([legacy, extension], ignore_index=True, sort=False)
    training_sec = training[training.source_family.eq("US_SEC")].reset_index(drop=True)
    prepared_sec = _assert_feature_equivalence(training, training_sec)

    numeric = v36.make_model("lightgbm", dict(v38.PARAMETERS))
    numeric.fit(
        scorer.numeric_feature_frame(training), training.y.astype(int),
        model__sample_weight=training.cluster_weight,
    )
    text = v59.TextExpert(v223.FROZEN_C).fit(prepared_sec)
    semantic_structure = v59.StructureExpert(v223.FROZEN_C).fit(prepared_sec)
    training_metadata = {
        "all_source_training_rows": int(len(training)),
        "us_sec_training_rows": int(len(training_sec)),
        "legacy_rows": int(len(legacy)),
        "extension_rows": int(len(extension)),
        "training_time_min_utc": pd.Timestamp(training.event_time_utc.min()).isoformat(),
        "training_time_max_utc": pd.Timestamp(training.event_time_utc.max()).isoformat(),
        "extension_time_max_utc": pd.Timestamp(extension.event_time_utc.max()).isoformat(),
        "data_epoch_sha256": str(epoch["data_epoch_sha256"]),
        "numeric_transformed_features": int(len(numeric.named_steps["features"].get_feature_names_out())),
        "text_vocabulary_features": int(len(text.vector.vocabulary_)),
        "semantic_structure_features": int(len(semantic_structure.vector.feature_names_)),
    }
    bundle = {
        "schema_version": scorer.SCHEMA_VERSION,
        "artifact_role": "FROZEN_RESEARCH_CONFIRMATION_SCORER_NOT_SEAL_OR_FINAL",
        "model_family": scorer.MODEL_FAMILY,
        "models": {
            "numeric_pipeline": numeric,
            "text_vectorizer": text.vector,
            "text_classifier": text.model,
            "semantic_structure_vectorizer": semantic_structure.vector,
            "semantic_structure_classifier": semantic_structure.model,
        },
        "frozen_algorithm": {
            "blend_weights": dict(scorer.BLEND_WEIGHTS),
            "probability_clip": scorer.PROBABILITY_CLIP,
            "polarity_auc_floor": scorer.POLARITY_AUC_FLOOR,
            "cutoff_grid": [float(value) for value in scorer.CUTOFF_GRID],
            "confidence_quantile": scorer.CONFIDENCE_QUANTILE,
            "direction_threshold": scorer.DIRECTION_THRESHOLD,
            "cost": scorer.COST,
            "confirmation_block_size": scorer.CONFIRMATION_BLOCK_SIZE,
        },
        "training_metadata": training_metadata,
    }
    return bundle, legacy, extension, training_metadata


def deterministic_pickle_gzip(value: Any) -> bytes:
    raw = pickle.dumps(value, protocol=5)
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as handle:
        handle.write(raw)
    return output.getvalue()


def build_initial_state() -> dict[str, Any]:
    raw = pd.read_csv(
        ROOT / "output_V223" / "V223_FROZEN_V59_N_T_S_DIAGNOSTIC_PREDICTIONS.csv.gz",
        compression="gzip", low_memory=False, parse_dates=["event_time_utc"],
    ).sort_values(["fold", "event_time_utc", "event_id"]).reset_index(drop=True)
    guarded = pd.read_csv(
        ROOT / "output_V224" / "V224_EXACT_SEC_N_T_S_GUARDED_PREDICTIONS.csv.gz",
        compression="gzip", low_memory=False, parse_dates=["event_time_utc"],
    ).sort_values(["fold", "event_time_utc", "event_id"]).reset_index(drop=True)
    if raw.event_id.astype(str).tolist() != guarded.event_id.astype(str).tolist():
        raise RuntimeError("V224_FREEZE_RAW_GUARDED_EVENT_ID_MISMATCH")
    last_fold = int(guarded.fold.max())
    previous = guarded[guarded.fold.eq(last_fold)].copy()
    state = {
        "schema_version": scorer.SCHEMA_VERSION,
        "artifact_role": "V224_PROSPECTIVE_CONFIRMATION_INITIAL_DEV_STATE",
        "status": "FROZEN",
        "model_family": scorer.MODEL_FAMILY,
        "research_status": "DEV_RESEARCH_SURVIVOR_PENDING_INDEPENDENT_CONFIRMATION",
        "confirmation_contract": {
            "role": "V224_PROSPECTIVE_CONFIRMATION",
            "target_n": 100,
            "preliminary_min_n": 60,
            "block_size": 20,
            "checkpoint_n": [20, 40, 60, 80, 100],
            "chronology_only": True,
            "research_seal_pool_used": False,
            "final_meta_reserve_used": False,
        },
        "machine_state": {
            "next_confirmation_block": 1,
            "confirmation_rows_completed": 0,
            "completed_confirmation_blocks": [],
            "cumulative_raw_history": {
                "origin": "OPENED_COMPLETED_V224_DEV_ONLY",
                "event_id": raw.event_id.astype(str).tolist(),
                "raw_probability": raw.prob.astype(float).tolist(),
                "y": raw.y.astype(int).tolist(),
            },
            "previous_completed_block": {
                "origin": f"OPENED_COMPLETED_V224_DEV_FOLD_{last_fold}",
                "event_id": previous.event_id.astype(str).tolist(),
                "oriented_probability": previous.oriented_prob.astype(float).tolist(),
                "guarded_probability": previous.prob.astype(float).tolist(),
                "prediction": previous.prediction.astype(bool).tolist(),
                "margin": previous.margin.astype(float).tolist(),
                "y": previous.y.astype(int).tolist(),
                "fwd_ret_30m": previous.fwd_ret_30m.astype(float).tolist(),
            },
        },
        "guards": {
            "confirmation_labels_seen": False,
            "confirmation_returns_seen": False,
            "research_seal_or_final_rows_read": False,
            "v224_algorithm_mutated": False,
        },
    }
    state["initial_next_block_policy"] = scorer.derive_policy(state)

    cumulative_auc = float(v224.roc_auc_score(raw.y, raw.prob))
    expected_cutoff = float(v224.select_cutoff(previous))
    expected_high = v224.confidence_score(previous, low=False)
    expected_low = v224.confidence_score(previous, low=True)
    policy = state["initial_next_block_policy"]
    if policy["prior_raw_auc"] != cumulative_auc or policy["direction_cutoff"] != expected_cutoff:
        raise RuntimeError("V224_FREEZE_INITIAL_DIRECTION_STATE_DRIFT")
    if policy["previous_high_margin_score"] != [int(expected_high[0]), float(expected_high[1])]:
        raise RuntimeError("V224_FREEZE_INITIAL_HIGH_MARGIN_STATE_DRIFT")
    if policy["previous_low_margin_score"] != [int(expected_low[0]), float(expected_low[1])]:
        raise RuntimeError("V224_FREEZE_INITIAL_LOW_MARGIN_STATE_DRIFT")
    return state


def feature_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in scorer.FORBIDDEN_SCORE_COLUMNS if column in frame], errors="ignore")


def build_artifacts() -> tuple[dict[str, bytes], dict[str, Any]]:
    if os.environ.get("PYTHONHASHSEED") != REQUIRED_PYTHONHASHSEED:
        raise RuntimeError("V224_FREEZE_REQUIRES_PYTHONHASHSEED_0")
    verify_pins()
    bundle, legacy, extension, training_metadata = fit_bundle()
    state = build_initial_state()
    model_bytes = deterministic_pickle_gzip(bundle)
    scorer_bytes = SCORER_SOURCE.read_bytes()

    sample = extension.sort_values(["event_time_utc", "event_id"]).head(scorer.CONFIRMATION_BLOCK_SIZE).copy()
    scored = scorer.score_block(feature_only(sample), bundle, state)
    self_test = {
        "purpose": "serialization_and_full_preprocessing_determinism_only_not_performance_evidence",
        "event_ids": scored.event_id.astype(str).tolist(),
        "input_event_identity_sha256": canonical_digest(scored.event_id.astype(str).tolist()),
        "scored_output_sha256": scorer.score_digest(scored),
        "state_fingerprint_before": scorer.state_fingerprint(state),
        "labels_or_returns_used_to_score": False,
    }
    spec = {
        "schema_version": scorer.SCHEMA_VERSION,
        "artifact_role": "FROZEN_V224_PROSPECTIVE_CONFIRMATION_RESEARCH_ARTIFACT",
        "official_research_seal_certificate": False,
        "final_meta_artifact": False,
        "version": 224,
        "status": "DEV_RESEARCH_SURVIVOR_PENDING_INDEPENDENT_CONFIRMATION",
        "primary_candidate": scorer.MODEL_FAMILY,
        "architecture": {
            "direction": "logit-space blend of numeric N, SEC segment-text T, structured-semantic S",
            "weights": dict(scorer.BLEND_WEIGHTS),
            "numeric_model": {"family": "LightGBM", "parameters": dict(v38.PARAMETERS),
                              "random_state": v36.SEED, "sample_weight": "cluster_weight"},
            "text_model": {"vectorizer": "word TF-IDF (1,2), min_df=3, max_df=.995, max_features=24000",
                           "classifier": "balanced liblinear logistic", "C": v223.FROZEN_C,
                           "max_iter": 350, "random_state": 59},
            "semantic_model": {"meaning": "explicit SEC structured-semantics expert",
                               "vectorizer": "DictVectorizer sparse float32",
                               "classifier": "balanced liblinear logistic", "C": v223.FROZEN_C,
                               "max_iter": 350, "random_state": 59},
            "numeric_preprocessing": "median imputation with missing indicators; source_family/event_type/form most-frequent + one-hot min_frequency=3",
            "sec_preprocessing": "frozen V59 segment extraction, taxonomy, item/exhibit and explicit numeric semantics",
        },
        "features": {
            "numeric_columns": list(scorer.NUMERIC_COLS),
            "categorical_columns": list(scorer.CAT_COLUMNS),
            "derived_numeric": {
                "event_clock": ["event_hour", "event_weekday", "event_month", "event_hour_sin", "event_hour_cos"],
                "absolute_returns": ["abs_pre_ret_2", "abs_pre_ret_5", "abs_pre_ret_15", "abs_pre_ret_30", "abs_pre_ret_60"],
                "return_slopes": ["ret_slope_2_15", "ret_slope_5_30"],
                "keyword_sentiment": "event_positive_kw - event_negative_kw - event_financing_kw",
                "market_relative": ["excess_ret_5", "excess_ret_15", "excess_ret_30", "excess_ret_60"],
            },
            "sec_taxonomy_names": sorted(scorer.TAXONOMY),
            "missing_numeric_policy": "frozen fitted median imputation; missing-indicator features preserved",
            "fitted_inventory": training_metadata,
        },
        "state_machine": {
            "polarity": "reverse current block iff cumulative strictly-prior raw AUC < 0.48",
            "calibration": "cutoff maximizing accuracy on immediately previous completed block",
            "cutoff_grid_start": 0.30, "cutoff_grid_stop_inclusive": 0.70, "cutoff_grid_step": 0.01,
            "confidence": "lexicographically choose high- vs low-margin 20% using immediately previous completed block accuracy/net score",
            "confidence_score": "(checks[accuracy>=.60, net>0], accuracy + 5*clip(net,-.01,.01))",
            "confidence_quantile": scorer.CONFIDENCE_QUANTILE,
            "cost": scorer.COST,
            "current_block_labels_or_returns_used": False,
            "update_boundary": "only after every outcome in the completed chronological block exists",
        },
        "source_and_provider_policy": {
            "accepted_source_family": ["US_SEC"],
            "exact_accession_body_required": True,
            "decision_time_information_only": True,
            "provider_acquisition_state_modified_by_freeze": False,
            "source_router_behavior": "reject non-US_SEC; other source specialists remain outside this scorer",
        },
        "event_contract": {
            "event_time": "t", "entry_target": "t+2m", "entry_slippage_max_seconds": 60,
            "exit_target": "actual entry +30m", "exit_slippage_max_seconds": 60,
            "hold_seconds": [1800, 1860], "regular_session_only": True,
            "causal_training_gap_minutes": 35,
        },
        "confirmation_contract": state["confirmation_contract"],
        "training": training_metadata,
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "numeric_threads": runtime_limits.THREAD_COUNT, "gpu_required": False,
            "python_hash_seed": int(REQUIRED_PYTHONHASHSEED),
            "random_seeds": {"numeric_lightgbm": v36.SEED, "text_logistic": 59,
                             "semantic_structure_logistic": 59},
        },
        "source_authority": {
            "v223_commit_sha256": PINNED["output_V223/COMMIT.json"],
            "v224_commit_sha256": PINNED["output_V224/COMMIT.json"],
            "v223_runner_sha256": PINNED["experiment_v223_exact_sec_semantic_transfer.py"],
            "v224_runner_sha256": PINNED["experiment_v224_causal_polarity_confidence_guard.py"],
            "new_data_epoch_sha256": training_metadata["data_epoch_sha256"],
        },
        "selection_boundary": {
            "independent_confirmation_complete": False, "eligible_for_research_seal": False,
            "eligible_for_final_meta": False, "v69_historical_champion_changed": False,
            "research_seal_or_final_rows_read": False,
        },
        "deterministic_self_test": self_test,
    }
    artifacts = {
        SCORER_NAME: scorer_bytes,
        MODEL_NAME: model_bytes,
        SPEC_NAME: json_bytes(spec),
        STATE_NAME: json_bytes(state),
    }
    source_dependencies = {
        **{relative: {"sha256": expected, "bytes": (ROOT / relative).stat().st_size}
           for relative, expected in sorted(PINNED.items())},
        "freeze_v224_deployable_scorer.py": {
            "sha256": sha256_file(Path(__file__)), "bytes": Path(__file__).stat().st_size,
        },
        "v224_deployable_scorer.py": {
            "sha256": sha256_file(SCORER_SOURCE), "bytes": SCORER_SOURCE.stat().st_size,
        },
        "test_v224_deployable_scorer_determinism.py": {
            "sha256": sha256_file(TEST_SOURCE), "bytes": TEST_SOURCE.stat().st_size,
        },
    }
    files = {name: {"sha256": sha256_bytes(value), "bytes": len(value)}
             for name, value in sorted(artifacts.items())}
    freeze_set = {"files": files, "source_dependencies": source_dependencies}
    sha_manifest = {
        "schema_version": 1,
        "artifact_role": "V224_PROSPECTIVE_CONFIRMATION_FREEZE_SHA256",
        "hash_algorithm": "SHA-256",
        "files": files,
        "source_dependencies": source_dependencies,
        "freeze_set_sha256": canonical_digest(freeze_set),
        "manifest_scope_excludes": [SHA_NAME],
        "research_seal_or_final_rows_read": False,
        "output_v223_or_v224_modified": False,
    }
    artifacts[SHA_NAME] = json_bytes(sha_manifest)
    summary = {
        "status": "PASS",
        "artifact_count": len(artifacts),
        "freeze_set_sha256": sha_manifest["freeze_set_sha256"],
        "model_sha256": files[MODEL_NAME]["sha256"],
        "scorer_sha256": files[SCORER_NAME]["sha256"],
        "spec_sha256": files[SPEC_NAME]["sha256"],
        "state_sha256": files[STATE_NAME]["sha256"],
        "initial_next_block_policy": state["initial_next_block_policy"],
        "self_test_sha256": self_test["scored_output_sha256"],
        "research_seal_or_final_rows_read": False,
    }
    return artifacts, summary


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def freeze(output_dir: Path) -> dict[str, Any]:
    if (output_dir / SHA_NAME).exists():
        raise RuntimeError("V224_FREEZE_ALREADY_EXISTS_USE_VERIFY_OR_EXPLICIT_REPLACE")
    artifacts, summary = build_artifacts()
    for name, value in artifacts.items():
        atomic_write(output_dir / name, value)
    summary["output_dir"] = str(output_dir.resolve())
    summary["output_written"] = True
    return summary


def verify_existing(output_dir: Path) -> dict[str, Any]:
    verify_pins()
    manifest_path = output_dir / SHA_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for name, record in manifest["files"].items():
        path = output_dir / name
        if not path.is_file():
            mismatches.append({"file": name, "reason": "MISSING"})
        elif path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            mismatches.append({"file": name, "reason": "HASH_OR_SIZE_MISMATCH"})
    for relative, record in manifest["source_dependencies"].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            mismatches.append({"file": relative, "reason": "SOURCE_DEPENDENCY_MISMATCH"})
    if mismatches:
        raise RuntimeError("V224_FROZEN_ARTIFACT_DETERMINISM_FAIL:" + json.dumps(mismatches, sort_keys=True))
    spec = json.loads((output_dir / SPEC_NAME).read_text(encoding="utf-8"))
    state = json.loads((output_dir / STATE_NAME).read_text(encoding="utf-8"))
    bundle = scorer.load_bundle(output_dir / MODEL_NAME)
    _, extension, _, _ = v223.load_data()
    wanted = spec["deterministic_self_test"]["event_ids"]
    indexed = extension.set_index(extension.event_id.astype(str), drop=False)
    sample = indexed.loc[wanted].reset_index(drop=True)
    state_before = scorer.state_fingerprint(state)
    first = scorer.score_block(feature_only(sample), bundle, state)
    second = scorer.score_block(feature_only(sample.sample(frac=1.0, random_state=224)), bundle, state)
    if not first.equals(second) or scorer.score_digest(first) != spec["deterministic_self_test"]["scored_output_sha256"]:
        raise RuntimeError("V224_FROZEN_SCORER_INFERENCE_DETERMINISM_FAIL")
    if state_before != scorer.state_fingerprint(state) or state_before != spec["deterministic_self_test"]["state_fingerprint_before"]:
        raise RuntimeError("V224_FROZEN_SCORE_MUTATED_INITIAL_STATE")
    summary = {
        "status": "PASS",
        "output_dir": str(output_dir.resolve()),
        "output_written": False,
        "artifact_hashes_match": True,
        "source_dependency_hashes_match": True,
        "deterministic_inference_matches": True,
        "self_test_sha256": scorer.score_digest(first),
        "freeze_set_sha256": manifest["freeze_set_sha256"],
        "research_seal_or_final_rows_read": False,
    }
    return summary


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != REQUIRED_PYTHONHASHSEED:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = REQUIRED_PYTHONHASHSEED
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
                                   cwd=ROOT, env=environment, check=False)
        return int(completed.returncode)
    parser = argparse.ArgumentParser(description="Freeze or verify the complete V224 deployable scorer")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    runtime_limits.configure()
    if args.verify_existing:
        result = verify_existing(args.output_dir)
    elif args.replace_existing:
        artifacts, result = build_artifacts()
        for name, value in artifacts.items():
            atomic_write(args.output_dir / name, value)
        result["output_dir"] = str(args.output_dir.resolve())
        result["output_written"] = True
        result["existing_freeze_replaced"] = True
    elif (args.output_dir / SHA_NAME).exists():
        result = verify_existing(args.output_dir)
    else:
        result = freeze(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
