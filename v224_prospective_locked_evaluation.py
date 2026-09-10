"""Prequential locked evaluation of V224 prospective 20-row checkpoints.

For every chronological block, the frozen scorer sees a label/outcome-free
projection first.  Its predictions and manifest are durably written before
this process opens that block's outcomes.  Only then is the frozen
``complete_block`` transition applied and stored as a new immutable state
snapshot.  No central role/queue/epoch, Research Seal, or Final Meta authority
is imported or opened.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
POLICY_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION_POLICY_V1.json")
CHECKPOINT_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION/checkpoints")
OUTPUT_REL = Path("research/V224_PROSPECTIVE_CONFIRMATION/locked_evaluation")
FROZEN_REL = Path("cert_research/V224_CONFIRMATION")
FROZEN_SHA_NAME = "FROZEN_V224_SHA256.json"
FROZEN_SCORER_NAME = "FROZEN_V224_DEPLOYABLE_SCORER.py"
FROZEN_MODEL_NAME = "FROZEN_V224_DEPLOYABLE_SCORER.pkl.gz"
FROZEN_SPEC_NAME = "FROZEN_V224_SPEC.json"
FROZEN_STATE_NAME = "FROZEN_V224_STATE.json"
PINNED_FROZEN_SHA256 = "06db4f18cead6628da2aafeea8a0eaa14c7f7656f60f5a22f7bc0b266a0ebcac"
PINNED_POLICY_SHA256 = "02457e2592b188741bd2054f5e1df111b3da6926f027c5ce14752ac44591ff2f"
CHECKPOINTS = (20, 40, 60, 80, 100)
BLOCK_SIZE = 20
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_REPLICATES = 2000
OUTCOME_COLUMNS = {"y", "fwd_ret_30m", "future_return", "correct", "correctness", "strategy_net", "model_correctness", "up_down"}


@dataclass(frozen=True)
class FrozenRuntime:
    scorer: Any
    bundle: Any
    initial_state: dict[str, Any]
    spec: dict[str, Any]
    freeze_manifest: dict[str, Any]
    freeze_manifest_sha256: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False, default=str,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False, default=str,
        )
        + "\n"
    ).encode("utf-8")


def canonical_records_sha256(frame: pd.DataFrame) -> str:
    return sha256_bytes(canonical_json_bytes(frame.to_dict(orient="records")))


def deterministic_csv_gzip_bytes(frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
            frame.to_csv(text_handle, index=False, lineterminator="\n")
    return output.getvalue()


def _atomic_replace(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_immutable(path: Path, value: bytes) -> str:
    """Create once; an existing byte difference is an integrity failure."""
    if path.exists():
        existing = path.read_bytes()
        if existing != value:
            raise RuntimeError(f"IMMUTABLE_ARTIFACT_MISMATCH:{path.name}")
    else:
        _atomic_replace(path, value)
    return sha256_bytes(value)


def write_convergent(path: Path, value: bytes) -> str:
    """Write a current status only when its deterministic bytes changed."""
    if not path.exists() or path.read_bytes() != value:
        _atomic_replace(path, value)
    return sha256_bytes(value)


def verify_policy(root: Path) -> tuple[dict[str, Any], str]:
    path = root / POLICY_REL
    actual = sha256_file(path)
    if actual != PINNED_POLICY_SHA256:
        raise RuntimeError(f"V224_PROSPECTIVE_POLICY_SHA_MISMATCH:{actual}")
    policy = json.loads(path.read_text(encoding="utf-8"))
    role = policy["role_contract"]
    decision = policy["decision_contract"]
    exact = policy["exact_price_contract"]
    if (
        role.get("new_role") != "V224_PROSPECTIVE_CONFIRMATION"
        or role.get("target_rows") != 100
        or role.get("preliminary_rows") != 60
        or role.get("checkpoint_block_rows") != 20
        or role.get("checkpoint_rows") != list(CHECKPOINTS)
        or exact.get("contract_version") != "EXACT_T2_30M_V36"
        or decision.get("preliminary_60_is_non_decisional") is not True
        or decision.get("checkpoint_results_may_change_protocol") is not False
        or decision.get("v224_results_may_change_protocol") is not False
        or decision.get("watcher_may_open_research_seal") is not False
        or decision.get("watcher_may_open_final_meta") is not False
    ):
        raise RuntimeError("V224_PROSPECTIVE_POLICY_CONTRACT_MISMATCH")
    return policy, actual


def verify_frozen_files(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = root / FROZEN_REL
    manifest_path = directory / FROZEN_SHA_NAME
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != PINNED_FROZEN_SHA256:
        raise RuntimeError(f"V224_FROZEN_SHA_MANIFEST_MISMATCH:{manifest_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for name, record in manifest.get("files", {}).items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            mismatches.append(name)
    freeze_set = {
        "files": manifest.get("files", {}),
        "source_dependencies": manifest.get("source_dependencies", {}),
    }
    if sha256_bytes(canonical_json_bytes(freeze_set)) != manifest.get("freeze_set_sha256"):
        mismatches.append("freeze_set_sha256")
    required = {
        FROZEN_SCORER_NAME, FROZEN_MODEL_NAME, FROZEN_SPEC_NAME, FROZEN_STATE_NAME,
    }
    if set(manifest.get("files", {})) != required:
        mismatches.append("frozen_file_inventory")
    if mismatches:
        raise RuntimeError("V224_FROZEN_FILE_GUARD_FAIL:" + ",".join(sorted(mismatches)))
    spec = json.loads((directory / FROZEN_SPEC_NAME).read_text(encoding="utf-8"))
    state = json.loads((directory / FROZEN_STATE_NAME).read_text(encoding="utf-8"))
    contract = spec.get("confirmation_contract", {})
    if (
        spec.get("primary_candidate") != "V224_EXACT_SEC_N_T_S_GUARDED"
        or spec.get("status") != "DEV_RESEARCH_SURVIVOR_PENDING_INDEPENDENT_CONFIRMATION"
        or contract.get("block_size") != 20
        or contract.get("checkpoint_n") != list(CHECKPOINTS)
        or contract.get("preliminary_min_n") != 60
        or contract.get("target_n") != 100
        or contract.get("research_seal_pool_used") is not False
        or contract.get("final_meta_reserve_used") is not False
        or state.get("machine_state", {}).get("confirmation_rows_completed") != 0
        or state.get("machine_state", {}).get("next_confirmation_block") != 1
    ):
        raise RuntimeError("V224_FROZEN_SPEC_OR_STATE_CONTRACT_MISMATCH")
    return manifest, spec, state


def load_frozen_runtime(root: Path = ROOT) -> FrozenRuntime:
    manifest, spec, state = verify_frozen_files(root)
    scorer_path = root / FROZEN_REL / FROZEN_SCORER_NAME
    module_spec = importlib.util.spec_from_file_location("_v224_frozen_locked_scorer", scorer_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("V224_FROZEN_SCORER_IMPORT_FAIL")
    scorer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(scorer)
    bundle = scorer.load_bundle(root / FROZEN_REL / FROZEN_MODEL_NAME)
    before = scorer.state_fingerprint(state)
    expected = spec["deterministic_self_test"]["state_fingerprint_before"]
    if before != expected:
        raise RuntimeError("V224_FROZEN_INITIAL_STATE_FINGERPRINT_MISMATCH")
    return FrozenRuntime(
        scorer=scorer,
        bundle=bundle,
        initial_state=state,
        spec=spec,
        freeze_manifest=manifest,
        freeze_manifest_sha256=PINNED_FROZEN_SHA256,
    )


def _checkpoint_paths(root: Path, cumulative_n: int) -> tuple[Path, Path]:
    directory = root / CHECKPOINT_REL
    return (
        directory / f"CHECKPOINT_{cumulative_n:03d}.json",
        directory / f"V224_PROSPECTIVE_EXACT_BLOCK_{cumulative_n:03d}.csv.gz",
    )


def available_checkpoints(root: Path) -> list[int]:
    present: list[int] = []
    gap_seen = False
    for cumulative_n in CHECKPOINTS:
        manifest_path, snapshot_path = _checkpoint_paths(root, cumulative_n)
        either = manifest_path.exists() or snapshot_path.exists()
        both = manifest_path.is_file() and snapshot_path.is_file()
        if either and not both:
            raise RuntimeError(f"INCOMPLETE_WATCHER_CHECKPOINT_PAIR:{cumulative_n}")
        if both:
            if gap_seen:
                raise RuntimeError(f"NONCONSECUTIVE_WATCHER_CHECKPOINT:{cumulative_n}")
            present.append(cumulative_n)
        else:
            gap_seen = True
    return present


def _feature_columns(header: list[str], scorer: Any) -> list[str]:
    required = ["event_id", "event_time_utc", "source_family", "headline", "body", "form", "event_type"]
    missing = [column for column in required if column not in header]
    if missing:
        raise RuntimeError("CHECKPOINT_SCORE_FEATURES_MISSING:" + ",".join(missing))
    return required + [column for column in scorer.NUMERIC_COLS if column in header and column not in required]


def prevalidate_checkpoint(
    root: Path,
    cumulative_n: int,
    policy_sha: str,
    scorer: Any,
    previous_last_time: pd.Timestamp | None,
    seen_ids: set[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.Timestamp, str, str]:
    manifest_path, snapshot_path = _checkpoint_paths(root, cumulative_n)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    expected_start = cumulative_n - BLOCK_SIZE + 1
    expected_snapshot_rel = str(snapshot_path.relative_to(root)).replace("\\", "/")
    if (
        manifest.get("track") != "V224_PROSPECTIVE_CONFIRMATION"
        or int(manifest.get("checkpoint_rows", -1)) != cumulative_n
        or int(manifest.get("rows", -1)) != BLOCK_SIZE
        or int(manifest.get("cumulative_rows", -1)) != cumulative_n
        or int(manifest.get("block_start_observation_order", -1)) != expected_start
        or int(manifest.get("block_end_observation_order", -1)) != cumulative_n
        or manifest.get("policy_sha256") != policy_sha
        or manifest.get("snapshot_path") != expected_snapshot_rel
        or manifest.get("source_family") != "US_SEC"
        or manifest.get("contract_version") != "EXACT_T2_30M_V36"
        or manifest.get("protocol_change_authority") is not False
        or manifest.get("seal_or_final_open_authority") is not False
        or manifest.get("v224_evaluation_run_by_watcher") is not False
        or manifest.get("snapshot_sha256") != sha256_file(snapshot_path)
    ):
        raise RuntimeError(f"WATCHER_CHECKPOINT_MANIFEST_GUARD_FAIL:{cumulative_n}")
    header = pd.read_csv(snapshot_path, nrows=0).columns.tolist()
    if not {"y", "fwd_ret_30m", "prospective_observation_order"}.issubset(header):
        raise RuntimeError(f"WATCHER_CHECKPOINT_OUTCOME_SCHEMA_MISSING:{cumulative_n}")
    features = pd.read_csv(snapshot_path, usecols=_feature_columns(header, scorer), low_memory=False)
    identity = pd.read_csv(
        snapshot_path,
        usecols=["prospective_observation_order", "event_id", "event_time_utc"],
        low_memory=False,
    )
    if len(features) != BLOCK_SIZE or len(identity) != BLOCK_SIZE:
        raise RuntimeError(f"WATCHER_CHECKPOINT_BLOCK_SIZE_FAIL:{cumulative_n}")
    orders = pd.to_numeric(identity.prospective_observation_order, errors="raise").astype(int).tolist()
    if orders != list(range(expected_start, cumulative_n + 1)):
        raise RuntimeError(f"WATCHER_CHECKPOINT_OBSERVATION_ORDER_FAIL:{cumulative_n}")
    times = pd.to_datetime(identity.event_time_utc, utc=True, errors="raise")
    keys = list(zip(times.tolist(), identity.event_id.astype(str).tolist()))
    if keys != sorted(keys):
        raise RuntimeError(f"WATCHER_CHECKPOINT_NOT_CHRONOLOGICAL:{cumulative_n}")
    if previous_last_time is not None and times.iloc[0] < previous_last_time:
        raise RuntimeError(f"WATCHER_CHECKPOINT_CROSS_BLOCK_CHRONOLOGY_FAIL:{cumulative_n}")
    ids = set(identity.event_id.astype(str))
    if len(ids) != BLOCK_SIZE or ids.intersection(seen_ids):
        raise RuntimeError(f"WATCHER_CHECKPOINT_DUPLICATE_EVENT_ID:{cumulative_n}")
    if set(features.columns).intersection(OUTCOME_COLUMNS):
        raise RuntimeError(f"OUTCOME_COLUMN_PROJECTED_AT_SCORE_TIME:{cumulative_n}")
    return (
        manifest,
        features,
        times.iloc[-1],
        sha256_bytes(manifest_bytes),
        sha256_file(snapshot_path),
    )


def _block_artifact_paths(root: Path, block_number: int) -> dict[str, Path]:
    directory = root / OUTPUT_REL / "blocks"
    return {
        "prediction": directory / f"BLOCK_{block_number:03d}_PREDICTIONS.csv.gz",
        "prediction_manifest": directory / f"BLOCK_{block_number:03d}_PREDICTIONS_MANIFEST.json",
        "state": directory / f"STATE_AFTER_BLOCK_{block_number:03d}.json",
        "completion": directory / f"BLOCK_{block_number:03d}_COMPLETION.json",
    }


def score_or_verify_prediction(
    root: Path,
    block_number: int,
    cumulative_n: int,
    features: pd.DataFrame,
    state_before: dict[str, Any],
    checkpoint_manifest_sha: str,
    snapshot_sha: str,
    runtime: FrozenRuntime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = _block_artifact_paths(root, block_number)
    state_before_sha = runtime.scorer.state_fingerprint(state_before)
    freshly_scored = runtime.scorer.score_block(features, runtime.bundle, state_before)
    prediction_bytes = deterministic_csv_gzip_bytes(freshly_scored)
    prediction_sha = sha256_bytes(prediction_bytes)
    relative_prediction = str(paths["prediction"].relative_to(root)).replace("\\", "/")
    manifest = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CONFIRMATION",
        "artifact_role": "IMMUTABLE_LABEL_BLIND_PREQUENTIAL_PREDICTION",
        "block": block_number,
        "block_rows": BLOCK_SIZE,
        "cumulative_n": cumulative_n,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "checkpoint_snapshot_sha256": snapshot_sha,
        "frozen_sha_manifest_sha256": runtime.freeze_manifest_sha256,
        "freeze_set_sha256": runtime.freeze_manifest["freeze_set_sha256"],
        "state_before_fingerprint": state_before_sha,
        "score_digest": runtime.scorer.score_digest(freshly_scored),
        "prediction_path": relative_prediction,
        "prediction_sha256": prediction_sha,
        "labels_or_returns_loaded_at_score_time": False,
        "prediction_persisted_before_outcome_open": True,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_authority_rows_read": 0,
        "protocol_change_authority": False,
    }
    manifest_bytes = pretty_json_bytes(manifest)
    if paths["prediction"].exists() != paths["prediction_manifest"].exists():
        raise RuntimeError(f"INCOMPLETE_IMMUTABLE_PREDICTION_PAIR:{block_number}")
    if paths["prediction"].exists():
        if paths["prediction"].read_bytes() != prediction_bytes:
            raise RuntimeError(f"IMMUTABLE_PREDICTION_LOGIC_MISMATCH:{block_number}")
        if paths["prediction_manifest"].read_bytes() != manifest_bytes:
            raise RuntimeError(f"IMMUTABLE_PREDICTION_MANIFEST_MISMATCH:{block_number}")
        # Exact byte equality to a fresh frozen-score execution is stronger
        # than round-tripping floats through CSV and digesting parsed dtypes.
        return freshly_scored, manifest
    # Both durable writes occur before _read_outcomes_after_prediction can run.
    write_immutable(paths["prediction"], prediction_bytes)
    write_immutable(paths["prediction_manifest"], manifest_bytes)
    return freshly_scored, manifest


def _read_outcomes_after_prediction(
    root: Path,
    block_number: int,
    cumulative_n: int,
    manifest: dict[str, Any],
    cumulative_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _block_artifact_paths(root, block_number)
    if not paths["prediction"].is_file() or not paths["prediction_manifest"].is_file():
        raise RuntimeError(f"OUTCOME_OPEN_BEFORE_PREDICTION_PERSIST:{block_number}")
    checkpoint_manifest_path, snapshot_path = _checkpoint_paths(root, cumulative_n)
    if sha256_file(paths["prediction"]) != manifest["prediction_sha256"]:
        raise RuntimeError(f"PREDICTION_CHANGED_BEFORE_OUTCOME_OPEN:{block_number}")
    full = pd.read_csv(snapshot_path, low_memory=False)
    if canonical_records_sha256(full) != json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))["block_records_sha256"]:
        raise RuntimeError(f"WATCHER_BLOCK_RECORD_HASH_MISMATCH:{cumulative_n}")
    cumulative = pd.concat([*cumulative_frames, full], ignore_index=True, sort=False)
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    if canonical_records_sha256(cumulative) != checkpoint_manifest["prefix_records_sha256"]:
        raise RuntimeError(f"WATCHER_PREFIX_RECORD_HASH_MISMATCH:{cumulative_n}")
    outcomes = full[["event_id", "y", "fwd_ret_30m"]].copy()
    if outcomes.isna().any().any():
        raise RuntimeError(f"INCOMPLETE_PROSPECTIVE_BLOCK_OUTCOMES:{cumulative_n}")
    return outcomes, full


def complete_or_verify_state(
    root: Path,
    block_number: int,
    cumulative_n: int,
    state_before: dict[str, Any],
    scored: pd.DataFrame,
    outcomes: pd.DataFrame,
    prediction_manifest: dict[str, Any],
    runtime: FrozenRuntime,
) -> dict[str, Any]:
    paths = _block_artifact_paths(root, block_number)
    next_state = runtime.scorer.complete_block(state_before, scored, outcomes)
    if int(next_state["machine_state"]["confirmation_rows_completed"]) != cumulative_n:
        raise RuntimeError(f"FROZEN_STATE_ROW_COUNT_MISMATCH:{block_number}")
    state_bytes = runtime.scorer.canonical_json_bytes(next_state)
    state_after_sha = runtime.scorer.state_fingerprint(next_state)
    write_immutable(paths["state"], state_bytes)
    completion = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CONFIRMATION",
        "artifact_role": "IMMUTABLE_COMPLETED_PREQUENTIAL_BLOCK",
        "block": block_number,
        "block_rows": BLOCK_SIZE,
        "cumulative_n": cumulative_n,
        "prediction_manifest_sha256": sha256_file(paths["prediction_manifest"]),
        "prediction_sha256": prediction_manifest["prediction_sha256"],
        "score_digest": prediction_manifest["score_digest"],
        "state_before_fingerprint": prediction_manifest["state_before_fingerprint"],
        "state_after_fingerprint": state_after_sha,
        "state_path": str(paths["state"].relative_to(root)).replace("\\", "/"),
        "state_sha256": sha256_file(paths["state"]),
        "outcomes_opened_after_prediction_persist": True,
        "state_transition": "FROZEN_SCORER_COMPLETE_BLOCK",
        "state_append_only": True,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_authority_rows_read": 0,
        "protocol_changed": False,
    }
    write_immutable(paths["completion"], pretty_json_bytes(completion))
    return next_state


def _as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false"]).all():
        raise RuntimeError("METRIC_BOOLEAN_PARSE_FAIL")
    return values.eq("true").to_numpy(bool)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or np.unique(y).size != 2:
        return None
    return float(roc_auc_score(y, score))


def _balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float | None:
    recalls = [
        float((prediction[y == label] == bool(label)).mean())
        for label in (0, 1)
        if (y == label).any()
    ]
    return float(np.mean(recalls)) if recalls else None


def confidence_bin_report(correct: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    if len(correct) == 0:
        return {
            "bins": [], "accuracy_correlation": None,
            "nondecreasing_fraction": None, "nondecreasing": None,
        }
    bin_count = min(5, len(correct))
    # First-ranked ties make bin membership deterministic without changing the
    # original confidence ordering.
    ranks = pd.Series(confidence).rank(method="first")
    bins = pd.qcut(ranks, q=bin_count, labels=False, duplicates="drop").astype(int).to_numpy()
    records: list[dict[str, Any]] = []
    for bin_number in sorted(np.unique(bins)):
        selected = bins == bin_number
        records.append({
            "bin": int(bin_number + 1),
            "n": int(selected.sum()),
            "confidence_min": float(np.min(confidence[selected])),
            "confidence_max": float(np.max(confidence[selected])),
            "accuracy": float(np.mean(correct[selected])),
        })
    accuracies = np.asarray([record["accuracy"] for record in records], float)
    correlation = None
    if len(accuracies) >= 2 and np.std(accuracies) > 0:
        correlation = float(np.corrcoef(np.arange(len(accuracies), dtype=float), accuracies)[0, 1])
    if len(accuracies) >= 2:
        nondecreasing_fraction = float(np.mean(np.diff(accuracies) >= 0))
        nondecreasing = bool(np.all(np.diff(accuracies) >= 0))
    else:
        nondecreasing_fraction = None
        nondecreasing = None
    return {
        "bins": records,
        "accuracy_correlation": correlation,
        "nondecreasing_fraction": nondecreasing_fraction,
        "nondecreasing": nondecreasing,
    }


def deterministic_block_bootstrap(
    scored_outcomes: pd.DataFrame,
    cost: float,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    frame = scored_outcomes.reset_index(drop=True).copy()
    if frame.empty:
        return {
            "n_boot": n_boot, "seed": seed, "resampling_unit": None,
            "resampling_groups": 0, "balanced_accuracy_lower95": None,
            "highconf_net_lower95": None,
        }
    if "event_time_utc" in frame:
        dates = pd.to_datetime(frame.event_time_utc, utc=True, errors="coerce").dt.date.astype(str)
    else:
        dates = pd.Series(["NaT"] * len(frame))
    if dates.ne("NaT").all():
        group_values = dates
        unit = "EVENT_DATE"
    elif "confirmation_block" in frame:
        group_values = frame.confirmation_block.astype(str)
        unit = "CONFIRMATION_BLOCK"
    else:
        group_values = pd.Series(["ALL_ROWS"] * len(frame))
        unit = "SINGLE_AVAILABLE_BLOCK"
    groups = [np.asarray(index, int) for index in frame.groupby(group_values, sort=True).groups.values()]
    rng = np.random.default_rng(seed)
    ba_values: list[float] = []
    hc_net_values: list[float] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in chosen])
        sample = frame.iloc[indices]
        y = pd.to_numeric(sample.y, errors="raise").astype(int).to_numpy()
        prediction = _as_bool(sample.prediction)
        ba = _balanced_accuracy(y, prediction)
        if ba is not None and np.isfinite(ba):
            ba_values.append(float(ba))
        high_conf = _as_bool(sample.high_conf)
        if high_conf.any():
            returns = pd.to_numeric(sample.fwd_ret_30m, errors="raise").to_numpy(float)
            net = np.where(prediction, 1.0, -1.0) * returns - cost
            value = float(np.mean(net[high_conf]))
            if np.isfinite(value):
                hc_net_values.append(value)
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "resampling_unit": unit,
        "resampling_groups": int(len(groups)),
        "balanced_accuracy_lower95": (
            float(np.quantile(ba_values, 0.025)) if ba_values else None
        ),
        "highconf_net_lower95": (
            float(np.quantile(hc_net_values, 0.025)) if hc_net_values else None
        ),
        "balanced_accuracy_valid_replicates": int(len(ba_values)),
        "highconf_net_valid_replicates": int(len(hc_net_values)),
    }


def metrics_for(
    scored_outcomes: pd.DataFrame,
    cost: float,
    include_bootstrap: bool = False,
) -> dict[str, Any]:
    y = pd.to_numeric(scored_outcomes.y, errors="raise").astype(int).to_numpy()
    returns = pd.to_numeric(scored_outcomes.fwd_ret_30m, errors="raise").to_numpy(float)
    prediction = _as_bool(scored_outcomes.prediction)
    high_conf = _as_bool(scored_outcomes.high_conf)
    probability = pd.to_numeric(scored_outcomes.probability, errors="raise").to_numpy(float)
    confidence = pd.to_numeric(scored_outcomes.confidence_signal, errors="raise").to_numpy(float)
    correct = prediction == y
    signed = np.where(prediction, 1.0, -1.0) * returns - cost
    hc_n = int(high_conf.sum())
    naive = float(max((y == 0).mean(), (y == 1).mean()))
    result = {
        "n": int(len(y)),
        "y_0": int((y == 0).sum()),
        "y_1": int((y == 1).sum()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": _balanced_accuracy(y, prediction),
        "auc": _safe_auc(y, probability),
        "naive_accuracy": naive,
        "edge_vs_naive": float(correct.mean() - naive),
        "all_trade_net": float(signed.mean()),
        "highconf_n": hc_n,
        "highconf_coverage": float(high_conf.mean()),
        "highconf_accuracy": float(correct[high_conf].mean()) if hc_n else None,
        "highconf_net": float(signed[high_conf].mean()) if hc_n else None,
        "confidence_correctness_auc": _safe_auc(correct.astype(int), confidence),
        "confidence_bin_monotonicity": confidence_bin_report(correct, confidence),
        "cost": float(cost),
    }
    if include_bootstrap:
        result["bootstrap"] = deterministic_block_bootstrap(scored_outcomes, cost)
    return result


def checkpoint_classification(n: int) -> dict[str, Any]:
    if n == 100:
        return {
            "classification": "FINAL_PROSPECTIVE_CONFIRMATION_REPORT",
            "preliminary": False,
            "final_prospective_checkpoint": True,
            "decision_authority": "PROSPECTIVE_CONFIRMATION_ONLY",
        }
    if n == 60:
        return {
            "classification": "PRELIMINARY_60_NON_DECISIONAL",
            "preliminary": True,
            "final_prospective_checkpoint": False,
            "decision_authority": "NONE",
        }
    return {
        "classification": "INTERIM_CHECKPOINT_NON_DECISIONAL",
        "preliminary": False,
        "final_prospective_checkpoint": False,
        "decision_authority": "NONE",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_checkpoint_reports(
    root: Path,
    cumulative_n: int,
    all_rows: pd.DataFrame,
    state: dict[str, Any],
    runtime: FrozenRuntime,
) -> dict[str, str]:
    directory = root / OUTPUT_REL
    aggregate = metrics_for(all_rows, float(runtime.scorer.COST), include_bootstrap=True)
    by_block = {
        str(block): metrics_for(group, float(runtime.scorer.COST))
        for block, group in all_rows.groupby("confirmation_block", sort=True)
    }
    classification = checkpoint_classification(cumulative_n)
    metrics = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CONFIRMATION",
        "model_family": "V224_EXACT_SEC_N_T_S_GUARDED",
        "cumulative_n": cumulative_n,
        "block_size": BLOCK_SIZE,
        "source_family": "US_SEC",
        "aggregate": aggregate,
        "by_block": by_block,
        "classification": classification,
        "frozen_sha_manifest_sha256": runtime.freeze_manifest_sha256,
        "freeze_set_sha256": runtime.freeze_manifest["freeze_set_sha256"],
        "state_fingerprint": runtime.scorer.state_fingerprint(state),
        "state_append_only": True,
        "labels_or_returns_used_to_score_current_block": False,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_authority_rows_read": 0,
        "protocol_changed": False,
        "protocol_change_authority": False,
        "research_seal_open_authority": False,
        "final_meta_open_authority": False,
    }
    metrics_path = directory / f"N{cumulative_n:03d}_METRICS.json"
    metrics_sha = write_immutable(metrics_path, pretty_json_bytes(metrics))
    lines = [
        f"# V224 Prospective Locked Evaluation - N={cumulative_n}",
        "",
        f"Status: `{classification['classification']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in (
        "n", "y_0", "y_1", "accuracy", "balanced_accuracy", "auc",
        "edge_vs_naive", "all_trade_net", "highconf_coverage",
        "highconf_accuracy", "highconf_net", "confidence_correctness_auc",
    ):
        lines.append(f"| {name} | {_fmt(aggregate[name])} |")
    lines.extend([
        f"| bootstrap_balanced_accuracy_lower95 | {_fmt(aggregate['bootstrap']['balanced_accuracy_lower95'])} |",
        f"| bootstrap_highconf_net_lower95 | {_fmt(aggregate['bootstrap']['highconf_net_lower95'])} |",
        f"| confidence_bin_accuracy_correlation | {_fmt(aggregate['confidence_bin_monotonicity']['accuracy_correlation'])} |",
        f"| confidence_bin_nondecreasing_fraction | {_fmt(aggregate['confidence_bin_monotonicity']['nondecreasing_fraction'])} |",
    ])
    lines.extend([
        "",
        "Predictions for every block were persisted before that block's outcomes were opened.",
        "This report cannot change the frozen protocol, champion, Research Seal, or Final Meta state.",
        "N=60 is preliminary and non-decisional; only N=100 is the final prospective-confirmation report.",
        "",
    ])
    markdown_path = directory / f"N{cumulative_n:03d}_REPORT.md"
    markdown_sha = write_immutable(markdown_path, "\n".join(lines).encode("utf-8"))
    status = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CONFIRMATION",
        "cumulative_n": cumulative_n,
        **classification,
        "metrics_path": str(metrics_path.relative_to(root)).replace("\\", "/"),
        "metrics_sha256": metrics_sha,
        "markdown_path": str(markdown_path.relative_to(root)).replace("\\", "/"),
        "markdown_sha256": markdown_sha,
        "state_fingerprint": runtime.scorer.state_fingerprint(state),
        "protocol_changed": False,
        "research_seal_opened": False,
        "final_meta_opened": False,
        "champion_changed": False,
    }
    status_path = directory / f"N{cumulative_n:03d}_STATUS.json"
    status_sha = write_immutable(status_path, pretty_json_bytes(status))
    return {
        "metrics_sha256": metrics_sha,
        "markdown_sha256": markdown_sha,
        "status_sha256": status_sha,
    }


def run(root: Path = ROOT, runtime: FrozenRuntime | None = None) -> dict[str, Any]:
    _, policy_sha = verify_policy(root)
    # Always verify the complete frozen byte set, including when a focused
    # test injects a lightweight compatible runtime.
    frozen_manifest, _, _ = verify_frozen_files(root)
    if runtime is None:
        runtime = load_frozen_runtime(root)
    elif runtime.freeze_manifest_sha256 != PINNED_FROZEN_SHA256:
        raise RuntimeError("INJECTED_RUNTIME_FREEZE_IDENTITY_MISMATCH")
    checkpoints = available_checkpoints(root)
    if not checkpoints:
        status = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CONFIRMATION",
            "phase": "LOCKED_PREQUENTIAL_EVALUATION",
            "status": "ZERO_EXACT_ROWS_NO_CHECKPOINT_READY",
            "available_checkpoint_rows": [],
            "completed_checkpoint_rows": [],
            "next_checkpoint_rows": 20,
            "one_shot": True,
            "poll_or_sleep": False,
            "exited_without_wait": True,
            "policy_sha256": policy_sha,
            "frozen_sha_manifest_sha256": PINNED_FROZEN_SHA256,
            "freeze_set_sha256": frozen_manifest["freeze_set_sha256"],
            "prediction_rows": 0,
            "prospective_outcome_rows_read": 0,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_authority_rows_read": 0,
            "state_append_only": True,
            "protocol_changed": False,
            "protocol_change_authority": False,
            "research_seal_open_authority": False,
            "final_meta_open_authority": False,
        }
        write_convergent(root / OUTPUT_REL / "EVALUATION_STATUS.json", pretty_json_bytes(status))
        return status

    state = runtime.initial_state
    if runtime.scorer.state_fingerprint(state) != runtime.spec["deterministic_self_test"]["state_fingerprint_before"]:
        raise RuntimeError("RUNTIME_INITIAL_STATE_FINGERPRINT_MISMATCH")
    previous_last_time: pd.Timestamp | None = None
    seen_ids: set[str] = set()
    cumulative_checkpoint_frames: list[pd.DataFrame] = []
    scored_outcome_frames: list[pd.DataFrame] = []
    report_hashes: dict[str, dict[str, str]] = {}
    for block_number, cumulative_n in enumerate(checkpoints, start=1):
        manifest, features, previous_last_time, checkpoint_manifest_sha, snapshot_sha = prevalidate_checkpoint(
            root, cumulative_n, policy_sha, runtime.scorer, previous_last_time, seen_ids,
        )
        seen_ids.update(features.event_id.astype(str))
        scored, prediction_manifest = score_or_verify_prediction(
            root, block_number, cumulative_n, features, state,
            checkpoint_manifest_sha, snapshot_sha, runtime,
        )
        outcomes, full_checkpoint = _read_outcomes_after_prediction(
            root, block_number, cumulative_n, prediction_manifest, cumulative_checkpoint_frames,
        )
        state = complete_or_verify_state(
            root, block_number, cumulative_n, state, scored, outcomes,
            prediction_manifest, runtime,
        )
        cumulative_checkpoint_frames.append(full_checkpoint)
        merged = scored.merge(outcomes, on="event_id", how="left", validate="one_to_one")
        scored_outcome_frames.append(merged)
        all_rows = pd.concat(scored_outcome_frames, ignore_index=True, sort=False)
        report_hashes[str(cumulative_n)] = write_checkpoint_reports(
            root, cumulative_n, all_rows, state, runtime,
        )
    completed_n = checkpoints[-1]
    classification = checkpoint_classification(completed_n)
    status = {
        "schema_version": 1,
        "track": "V224_PROSPECTIVE_CONFIRMATION",
        "phase": "LOCKED_PREQUENTIAL_EVALUATION",
        "status": classification["classification"],
        "available_checkpoint_rows": checkpoints,
        "completed_checkpoint_rows": checkpoints,
        "next_checkpoint_rows": next((value for value in CHECKPOINTS if value > completed_n), None),
        "one_shot": True,
        "poll_or_sleep": False,
        "exited_without_wait": True,
        "policy_sha256": policy_sha,
        "frozen_sha_manifest_sha256": PINNED_FROZEN_SHA256,
        "freeze_set_sha256": frozen_manifest["freeze_set_sha256"],
        "prediction_rows": completed_n,
        "prospective_outcome_rows_read": completed_n,
        "prediction_persisted_before_each_block_outcome_open": True,
        "state_fingerprint": runtime.scorer.state_fingerprint(state),
        "state_append_only": True,
        "report_hashes": report_hashes,
        **classification,
        "research_seal_rows_read": 0,
        "final_meta_rows_read": 0,
        "central_authority_rows_read": 0,
        "protocol_changed": False,
        "protocol_change_authority": False,
        "research_seal_open_authority": False,
        "final_meta_open_authority": False,
        "champion_changed": False,
    }
    write_convergent(root / OUTPUT_REL / "EVALUATION_STATUS.json", pretty_json_bytes(status))
    return status


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=ROOT, env=environment, check=False,
        )
        return int(completed.returncode)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    try:
        status = run(root)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "track": "V224_PROSPECTIVE_CONFIRMATION",
            "phase": "LOCKED_PREQUENTIAL_EVALUATION",
            "status": "FAILED_CLOSED",
            "error_type": type(error).__name__,
            "error": str(error),
            "one_shot": True,
            "poll_or_sleep": False,
            "research_seal_rows_read": 0,
            "final_meta_rows_read": 0,
            "central_authority_rows_read": 0,
            "protocol_changed": False,
            "protocol_change_authority": False,
        }
        write_convergent(root / OUTPUT_REL / "FAILED_CLOSED_STATUS.json", pretty_json_bytes(failure))
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
