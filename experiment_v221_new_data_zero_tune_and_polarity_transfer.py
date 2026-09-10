"""V221 new-data zero-tune and frozen-polarity transfer scaffold.

This module is deliberately not registered.  It cannot fit a model, choose a
threshold, choose an alpha, or read a research-seal/final-reserve row.  A real
evaluation becomes possible only after ``state/CURRENT_DATA_EPOCH.json`` and
the frozen activation contract both say READY.  Even then, predictions must
come from a separately frozen, label-blind scoring bundle; the V69 artifacts
contain OOF evidence, not a deployable fitted model, so reconstructing V69 by
training here would violate the requested zero-tune contract.

All current preparation modes are read-only and keep GPU visibility disabled.
The only output is a JSON document on stdout.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
VERSION = "V221"
HYPOTHESIS = "NEW_DATA_ZERO_TUNE_AND_POLARITY_TRANSFER"
STATE_PATH = ROOT / "state" / "CURRENT_DATA_EPOCH.json"
ACTIVATION_PATH = ROOT / "research" / "NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json"
READINESS_PATH = ROOT / "research" / "V70_NEW_DATA_EPOCH_READINESS.json"
EXTENSION_PATH = ROOT / "data" / "V59_DEV_EXTENSION_LABELED.csv.gz"
EXTENSION_MANIFEST_PATH = ROOT / "data" / "V59_DEV_EXTENSION_MANIFEST.json"
PREDICTION_ROOT = ROOT / "research" / "zero_tune_inputs" / "V221"

READY_STATES = {"READY", "NEW_DATA_EPOCH_READY"}
ALLOWED_ROLE = "DEV_EXTENSION"
ALLOWED_SOURCE_FAMILIES = {"US_SEC", "KR_NEWS"}
FORBIDDEN_PATH_TERMS = ("research_seal", "final_meta", "final_reserve", "untouched_pool")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COST = 0.002
CPU_MASK = 0xC0000000
NUMERIC_THREADS = "2"

PINNED_V69 = {
    "output_V69/COMMIT.json": "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31",
    "output_V69/ARTIFACT_MANIFEST.json": "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836",
    "output_V69/V69_FAIL_CLOSED_SELECTED_OOF.csv.gz": "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002",
}

COMPONENT_HOOKS = {
    "V69_LOCKED_CHAMPION": {
        "version": 69,
        "runner": "experiment_v69_us_sec_structured_agreement_selector.py",
        "runner_sha256": "86be2705c7c1b18241605350122d86483b10e4a7aa875dd8d629a78c72fc7136",
        "prediction_column": "V69_prob",
        "confidence_column": "V69_confidence_signal",
        "high_conf_column": "V69_high_conf",
        "primary_axes": ["direction", "confidence", "economic", "opportunity"],
        "required": True,
    },
    "V114_ZERO_TUNE": {
        "version": 114,
        "runner": "experiment_v114_duplicate_weighted_nonnegative_parts_factor.py",
        "runner_sha256": "596c4739abef8c2dad22ba9aab686dc322519fba4a75ce610edec443e154238e",
        "prediction_column": "V114_prob",
        "primary_axes": ["direction", "confidence", "economic"],
        "required": False,
    },
    "V126_ZERO_TUNE": {
        "version": 126,
        "runner": "experiment_v126_multichannel_gramian_angular_field_direction.py",
        "runner_sha256": "cbf5a2aa68a4383fa961489e50ea43d69244fb7ada51f2c475121e775ed6c74e",
        "prediction_column": "V126_prob",
        "primary_axes": ["direction", "confidence", "economic"],
        "required": False,
    },
    "V158_ZERO_TUNE": {
        "version": 158,
        "runner": "experiment_v158_causal_multiscale_event_arrival_hazard_direction.py",
        "runner_sha256": "8508675727658e91e09d5b1ecbb0615b9533769a93a9a061fe086c60218762de",
        "prediction_column": "V158_prob",
        "primary_axes": ["direction"],
        "required": False,
    },
    "V168_ZERO_TUNE": {
        "version": 168,
        "runner": "experiment_v168_multichannel_causal_class_conditional_matrix_normal_direction.py",
        "runner_sha256": "dba61c569ac78be67b6328a13358400e798f7f610332e9e138b4025d5d49ff34",
        "prediction_column": "V168_prob",
        "primary_axes": ["direction_auc"],
        "required": False,
    },
    "V203_ZERO_TUNE": {
        "version": 203,
        "runner": "experiment_v203_multichannel_causal_class_conditional_potts_grid_direction.py",
        "runner_sha256": "0de26adc064a2508ee476e249e7fcb4465c2720667e85983bd52e01275bccb68",
        "prediction_column": "V203_prob",
        "primary_axes": ["direction", "economic_robustness"],
        "required": False,
    },
}

OLD_POLARITY_HOOKS = {
    "V64_FEATURE_BLOCK_POLARITY": {
        "source_versions": [64],
        "reference_files": [
            "FEATURE_BLOCK_POLARITY_AUDIT.csv",
            "SOURCE_POLARITY_AUDIT.csv",
            "CONFIDENCE_RELIABILITY_REPORT.json",
            "EXTREME_REVERSAL_REPORT.json",
        ],
        "roles": ["DIRECTION", "CONFIDENCE"],
    },
    "V65_US_NEWS_CONFIDENCE_REVERSAL": {
        "source_versions": [65],
        "reference_files": ["US_NEWS_CONFIDENCE_REVERSAL_REPORT.json"],
        "roles": ["CONFIDENCE"],
        "source_scope": ["US_NEWS"],
    },
    "V66_EXTREME_REVERSAL_GUARD": {
        "source_versions": [66],
        "reference_files": ["EXTREME_REVERSAL_GUARD_REPORT.json"],
        "roles": ["CONFIDENCE"],
        "source_scope": ["US_NEWS"],
    },
    "V67_DIRECTION_OPPORTUNITY_POLARITY": {
        "source_versions": [67],
        "reference_files": ["DIRECTION_OPPORTUNITY_POLARITY_MAP_V67.json"],
        "roles": ["DIRECTION", "OPPORTUNITY"],
    },
    "V68_KR_NEWS_FROZEN_ROUTER": {
        "source_versions": [68],
        "reference_files": ["KR_NEWS_DIRECTION_OPPORTUNITY_ROUTER_REPORT.json"],
        "roles": ["DIRECTION", "OPPORTUNITY"],
        "source_scope": ["KR_NEWS"],
    },
    "V69_EXPERT_AGREEMENT": {
        "source_versions": [69],
        "reference_files": ["US_SEC_STRUCTURED_AGREEMENT_SELECTOR_REPORT.json"],
        "roles": ["CONFIDENCE"],
        "source_scope": ["US_SEC"],
    },
}

OLD_OUTPUT_PINS = {
    64: ("d8445dab8d0daceee2d951fe7174a25ebdfb087e1149f1b50f16038cbd5156f8", "ce7c719bcceaf6f0ecb553c07cc45f9224209dca170d24b464eeb46201370dd1"),
    65: ("18eafa9c482bc4743f5f14ae0829ac1c692ce5dd186e4e659f7002147b6cf5a7", "f6386e57b29d2e37d31d008e7891827380bbcb7db36abbf96cad9f8d7c472578"),
    66: ("336d6f38a3b41b0414d3078521f768d506207f09e051b5fa2d50dde4653be811", "74198a5011a587d7ba2d60ecbffaac7aa78a33386ec3da3beb49a1f40293986f"),
    67: ("de29762eb7a6cbf32fb645313da24fa99c175a7f894f70396d78f2437f266802", "8273a7cb933471bc1e2228ee7dc55b8de4eaaff01e4e5a193e21dc55f6fae865"),
    68: ("d6d9ff14b05a580dedb7aa5a8cc4891be201b5dee2261e76b3551eb5bee1150f", "fcaa9472313e0a4ae6765e37cbd55ce0f3f888e3825935beabbf85fa4ee0eea8"),
    69: (PINNED_V69["output_V69/COMMIT.json"], PINNED_V69["output_V69/ARTIFACT_MANIFEST.json"]),
}


class ContractError(RuntimeError):
    """Fail-closed contract violation."""


class EpochNotReady(ContractError):
    """The only expected state before the new exact epoch freezes."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required JSON missing: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path.name}")
    return value


def configure_bounded_resources() -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[name] = NUMERIC_THREADS
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
    observed: list[int] = []
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
        ]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        require(bool(kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(CPU_MASK))),
                "could not pin bounded V221 process to CPU30-31")
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        require(bool(kernel32.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask))),
                "could not verify V221 process affinity")
        require(process_mask.value == CPU_MASK, "V221 bounded affinity is not exactly CPU30-31")
        observed = [index for index in range(64) if process_mask.value & (1 << index)]
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {30, 31})
        observed = sorted(os.sched_getaffinity(0))
    return {
        "cpu_ids": observed,
        "affinity_mask_hex": f"0x{CPU_MASK:X}",
        "numeric_threads": 2,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_model_calls": 0,
    }


def inspect_epoch_state() -> dict[str, Any]:
    state = read_json(STATE_PATH)
    return {
        "status": state.get("status"),
        "activation_contract_written": bool(state.get("activation_contract_written", False)),
        "seal_or_final_labels_opened": bool(state.get("seal_or_final_labels_opened", False)),
        "blocking_reasons": list(state.get("blocking_reasons", [])),
        "checks": dict(state.get("checks", {})),
        "counts": state.get("counts", {}),
        "state_sha256": sha256(STATE_PATH),
    }


def require_ready_epoch() -> tuple[dict[str, Any], dict[str, Any]]:
    state = read_json(STATE_PATH)
    if state.get("status") not in READY_STATES:
        raise EpochNotReady(
            f"CURRENT_DATA_EPOCH is {state.get('status')!r}; V221 evaluation is blocked"
        )
    require(state.get("activation_contract_written") is True,
            "READY state lacks activation_contract_written=true")
    require(state.get("seal_or_final_labels_opened") is False,
            "seal/final labels were opened; V221 refuses the epoch")
    require(not state.get("blocking_reasons"), "READY state still has blocking reasons")
    checks = state.get("checks", {})
    require(checks and all(value is True for value in checks.values()),
            "READY state contains a failed readiness check")

    activation = read_json(ACTIVATION_PATH)
    require(activation.get("status") == "FROZEN" and activation.get("frozen") is True,
            "activation contract is not FROZEN")
    require(activation.get("role") == ALLOWED_ROLE,
            "activation contract is not DEV_EXTENSION-only")
    require(activation.get("selection_uses_labels") is False,
            "role selection used labels")
    require(activation.get("data_contract", {}).get("role") == ALLOWED_ROLE,
            "activation data contract role mismatch")
    forbidden = set(activation.get("data_contract", {}).get("reserved_roles_forbidden", []))
    require({"RESEARCH_SEAL_POOL", "FINAL_META_RESERVE"}.issubset(forbidden),
            "activation contract does not explicitly forbid reserved roles")
    epoch_digest = str(activation.get("data_epoch_sha256", ""))
    require(bool(HEX64.fullmatch(epoch_digest)), "activation data_epoch_sha256 is not frozen")
    state_digest = state.get("data_epoch_sha256")
    if state_digest is not None:
        require(state_digest == epoch_digest, "state/activation epoch digest mismatch")
    return state, activation


def exact_relative(path: Path, expected: Path) -> None:
    require(path.resolve() == expected.resolve(), f"unauthorized data path: {path}")


def verify_epoch_files(activation: dict[str, Any]) -> dict[str, str]:
    files = activation.get("files", {})
    specs = (
        ("extension_csv", EXTENSION_PATH),
        ("extension_manifest", EXTENSION_MANIFEST_PATH),
    )
    verified: dict[str, str] = {}
    for key, expected_path in specs:
        spec = files.get(key, {})
        candidate = ROOT / str(spec.get("path", ""))
        exact_relative(candidate, expected_path)
        require(expected_path.is_file(), f"epoch file missing: {expected_path.name}")
        actual = sha256(expected_path)
        require(actual == spec.get("sha256"), f"epoch file hash mismatch: {expected_path.name}")
        require(expected_path.stat().st_size == int(spec.get("bytes", -1)),
                f"epoch file size mismatch: {expected_path.name}")
        verified[str(expected_path.relative_to(ROOT)).replace("\\", "/")] = actual
    return verified


def verify_atomic_output(version: int, required_files: Iterable[str] = ()) -> dict[str, Any]:
    output = ROOT / f"output_V{version}"
    commit_path = output / "COMMIT.json"
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    expected_commit, expected_manifest = OLD_OUTPUT_PINS[version]
    require(sha256(commit_path) == expected_commit, f"V{version} COMMIT changed")
    require(sha256(manifest_path) == expected_manifest, f"V{version} manifest changed")
    manifest = read_json(manifest_path)
    entries = manifest.get("files", {})
    require(isinstance(entries, dict) and entries, f"V{version} manifest is empty")
    for name in required_files:
        require(name in entries, f"V{version} manifest lacks {name}")
    for name, spec in entries.items():
        path = output / name
        require(path.is_file(), f"V{version} manifest file missing: {name}")
        require(sha256(path) == spec.get("sha256"), f"V{version} artifact hash mismatch: {name}")
        require(path.stat().st_size == int(spec.get("bytes", -1)),
                f"V{version} artifact size mismatch: {name}")
    commit = read_json(commit_path)
    require(str(commit.get("version")) in {str(version), f"V{version}"},
            f"V{version} commit version mismatch")
    return {"commit_sha256": expected_commit, "manifest_sha256": expected_manifest,
            "manifest_files_verified": len(entries)}


def verify_frozen_authority() -> dict[str, Any]:
    v69 = verify_atomic_output(69, ("V69_FAIL_CLOSED_SELECTED_OOF.csv.gz",))
    for relative, expected in PINNED_V69.items():
        require(sha256(ROOT / relative) == expected, f"pinned V69 input changed: {relative}")
    components: dict[str, str] = {}
    for name, hook in COMPONENT_HOOKS.items():
        runner = ROOT / hook["runner"]
        require(runner.is_file(), f"hook runner missing: {runner.name}")
        actual = sha256(runner)
        if "runner_sha256" in hook:
            require(actual == hook["runner_sha256"], f"hook runner changed: {runner.name}")
        components[name] = actual
    old_outputs: dict[str, Any] = {}
    for probe, hook in OLD_POLARITY_HOOKS.items():
        version = int(hook["source_versions"][0])
        old_outputs[probe] = verify_atomic_output(version, hook["reference_files"])
    return {"v69": v69, "component_runner_sha256": components,
            "old_polarity_authority": old_outputs}


def load_extension(activation: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verified = verify_epoch_files(activation)
    required = {
        "event_id", "event_group_id", "market", "ticker", "event_time_utc",
        "source_family", "contract_version", "price_cadence_sec",
        "bar_timestamp_semantics", "fwd_ret_30m", "y", "data_role",
    }
    rows: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    seen_groups: set[str] = set()
    sources: Counter[str] = Counter()
    classes: dict[str, Counter[int]] = defaultdict(Counter)
    with gzip.open(EXTENSION_PATH, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                f"DEV_EXTENSION schema missing: {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            event_id, group_id = raw["event_id"], raw["event_group_id"]
            require(event_id not in seen_events and group_id not in seen_groups,
                    "DEV_EXTENSION repeats event_id or event_group_id")
            seen_events.add(event_id); seen_groups.add(group_id)
            require(raw["data_role"] == ALLOWED_ROLE,
                    "V221 refuses any non-DEV_EXTENSION row")
            require(raw["source_family"] in ALLOWED_SOURCE_FAMILIES,
                    f"unauthorized DEV_EXTENSION source: {raw['source_family']}")
            require(raw["contract_version"] == "EXACT_T2_30M_V36",
                    "DEV_EXTENSION exact contract mismatch")
            require(raw["bar_timestamp_semantics"] == "OPEN" and int(float(raw["price_cadence_sec"])) <= 60,
                    "DEV_EXTENSION bar semantics mismatch")
            y = int(raw["y"]); ret = float(raw["fwd_ret_30m"])
            require(y in (0, 1) and y == int(ret > 0.0), "DEV_EXTENSION label/return mismatch")
            rows.append({
                "event_id": event_id, "event_group_id": group_id,
                "source_family": raw["source_family"], "market": raw["market"],
                "ticker": raw["ticker"], "event_time_utc": raw["event_time_utc"],
                "event_date": raw["event_time_utc"][:10], "y": y,
                "fwd_ret_30m": ret,
            })
            sources[raw["source_family"]] += 1
            classes[raw["source_family"]][y] += 1
    require(set(sources) == ALLOWED_SOURCE_FAMILIES, "DEV_EXTENSION source set mismatch")
    for source in ALLOWED_SOURCE_FAMILIES:
        require(sources[source] >= 120 and classes[source][0] >= 25 and classes[source][1] >= 25,
                f"DEV_EXTENSION support below frozen minimum: {source}")
    expected_counts = activation.get("counts", {}).get("by_source_family", {})
    for source, count in sources.items():
        require(int(expected_counts.get(source, {}).get("n", -1)) == count,
                f"activation/CSV count mismatch: {source}")
    return rows, {"rows": len(rows), "sources": dict(sources),
                  "classes": {key: dict(value) for key, value in classes.items()},
                  "verified_files": verified, "roles_opened": [ALLOWED_ROLE],
                  "seal_or_final_labels_opened": False}


def path_has_forbidden_role(path: Path) -> bool:
    lowered = str(path).replace("\\", "/").lower()
    return any(term in lowered for term in FORBIDDEN_PATH_TERMS)


def load_prediction_bundle(
    manifest_path: Path, rows: list[dict[str, Any]], activation: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    require(not path_has_forbidden_role(manifest_path), "prediction manifest path names a forbidden role")
    require(manifest_path.is_relative_to(PREDICTION_ROOT.resolve()),
            "prediction manifest must be under research/zero_tune_inputs/V221")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == "market_bio_v221_zero_tune_bundle_v1",
            "zero-tune bundle schema mismatch")
    require(manifest.get("status") == "FROZEN_LABEL_BLIND", "prediction bundle is not frozen label-blind")
    require(manifest.get("data_epoch_sha256") == activation.get("data_epoch_sha256"),
            "prediction bundle epoch mismatch")
    invariant_flags = (
        "fit_executed_for_epoch", "retrained", "threshold_changed", "polarity_changed",
        "hyperparameters_changed", "dev_labels_read_by_scorer", "seal_or_final_rows_read",
        "outer_labels_used_for_selection",
    )
    for flag in invariant_flags:
        require(manifest.get(flag) is False, f"zero-tune invariant violated: {flag}")
    require(manifest.get("v69_commit_sha256") == PINNED_V69["output_V69/COMMIT.json"],
            "prediction bundle V69 authority mismatch")
    require(manifest.get("extension_sha256") == sha256(EXTENSION_PATH),
            "prediction bundle extension mismatch")
    hooks = manifest.get("hooks", {})
    require(hooks.get("V69_LOCKED_CHAMPION", {}).get("available") is True,
            "locked V69 predictions are mandatory")
    for name, hook_manifest in hooks.items():
        require(name in COMPONENT_HOOKS, f"unknown zero-tune hook: {name}")
        require(hook_manifest.get("fit_executed_for_epoch") is False,
                f"{name} was fit for the new epoch")
        require(hook_manifest.get("selection_or_tuning_executed") is False,
                f"{name} used selection or tuning")
        frozen_hash = str(hook_manifest.get("frozen_scorer_sha256", ""))
        require(bool(HEX64.fullmatch(frozen_hash)), f"{name} lacks a frozen scorer hash")

    prediction_spec = manifest.get("prediction_csv", {})
    prediction_path = (ROOT / str(prediction_spec.get("path", ""))).resolve()
    require(not path_has_forbidden_role(prediction_path), "prediction CSV path names a forbidden role")
    require(prediction_path.is_relative_to(PREDICTION_ROOT.resolve()),
            "prediction CSV must be under research/zero_tune_inputs/V221")
    require(prediction_path.is_file(), "prediction CSV missing")
    require(sha256(prediction_path) == prediction_spec.get("sha256"), "prediction CSV hash mismatch")

    available = {name: hook for name, hook in hooks.items() if hook.get("available") is True}
    required_columns = {"event_id"}
    for name in available:
        required_columns.add(COMPONENT_HOOKS[name]["prediction_column"])
    required_columns.update({"V69_confidence_signal", "V69_high_conf"})
    predictions: dict[str, dict[str, str]] = {}
    opener = gzip.open if prediction_path.name.endswith(".gz") else open
    with opener(prediction_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and required_columns.issubset(reader.fieldnames),
                f"prediction schema missing: {sorted(required_columns - set(reader.fieldnames or []))}")
        for record in reader:
            event_id = record["event_id"]
            require(event_id not in predictions, "prediction bundle repeats event_id")
            predictions[event_id] = record
    expected_ids = {row["event_id"] for row in rows}
    require(set(predictions) == expected_ids, "prediction/DEV_EXTENSION event set mismatch")
    return predictions, {"manifest_sha256": sha256(manifest_path),
                         "prediction_sha256": sha256(prediction_path),
                         "available_hooks": sorted(available)}


def safe_auc(target: list[int], score: list[float]) -> float | None:
    positives = sum(target); negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(score)), key=lambda index: score[index])
    ranks = [0.0] * len(score)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and score[order[stop]] == score[order[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    rank_sum = sum(rank for rank, y in zip(ranks, target) if y == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def stable_top_mask(rows: list[dict[str, Any]], score: list[float], fraction: float) -> list[bool]:
    count = max(1, int(math.ceil(len(rows) * fraction)))
    chosen = sorted(range(len(rows)), key=lambda index: (-score[index], rows[index]["event_id"]))[:count]
    selected = set(chosen)
    return [index in selected for index in range(len(rows))]


def summarize(rows: list[dict[str, Any]], probability: list[float], confidence: list[float],
              high_conf: list[bool]) -> dict[str, Any]:
    require(len(rows) == len(probability) == len(confidence) == len(high_conf) and rows,
            "metric vector length mismatch")
    require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probability + confidence),
            "non-finite or out-of-range score")
    y = [int(row["y"]) for row in rows]
    pred = [int(value >= 0.5) for value in probability]
    correct = [int(a == b) for a, b in zip(y, pred)]
    accuracy = sum(correct) / len(correct)
    recalls = []
    for label in (0, 1):
        indices = [index for index, actual in enumerate(y) if actual == label]
        recalls.append(sum(correct[index] for index in indices) / len(indices) if indices else float("nan"))
    balanced = sum(recalls) / 2.0 if all(math.isfinite(value) for value in recalls) else None
    naive = max(sum(y), len(y) - sum(y)) / len(y)
    signed_net = [(1.0 if prediction else -1.0) * row["fwd_ret_30m"] - COST
                  for row, prediction in zip(rows, pred)]
    hc_indices = [index for index, selected in enumerate(high_conf) if selected]
    hc_net = [signed_net[index] for index in hc_indices]
    top1 = sorted(hc_net, reverse=True)[1:] if len(hc_net) > 1 else []
    top5 = sorted(hc_net, reverse=True)[min(5, max(0, len(hc_net) - 1)):] if len(hc_net) > 1 else []
    if hc_net:
        sorted_hc = sorted(hc_net)
        lo = sorted_hc[int(0.01 * (len(sorted_hc) - 1))]
        hi = sorted_hc[int(0.99 * (len(sorted_hc) - 1))]
        winsorized = [min(hi, max(lo, value)) for value in hc_net]
    else:
        winsorized = []
    top_accuracy: dict[str, Any] = {}
    for fraction in (0.10, 0.20):
        mask = stable_top_mask(rows, confidence, fraction)
        indices = [index for index, selected in enumerate(mask) if selected]
        top_accuracy[str(fraction)] = sum(correct[index] for index in indices) / len(indices)
    return {
        "n": len(rows), "auc": safe_auc(y, probability), "balanced_accuracy": balanced,
        "accuracy": accuracy, "edge_vs_naive": accuracy - naive,
        "confidence_correctness_auc": safe_auc(correct, confidence),
        "highconf_coverage": len(hc_indices) / len(rows), "highconf_n": len(hc_indices),
        "highconf_accuracy": (sum(correct[index] for index in hc_indices) / len(hc_indices)) if hc_indices else None,
        "highconf_net": (sum(hc_net) / len(hc_net)) if hc_net else None,
        "winsorized_highconf_net": (sum(winsorized) / len(winsorized)) if winsorized else None,
        "top1_removed_highconf_net": (sum(top1) / len(top1)) if top1 else None,
        "top5_removed_highconf_net": (sum(top5) / len(top5)) if top5 else None,
        "top_accuracy": top_accuracy,
    }


def date_block_bootstrap_lower(rows: list[dict[str, Any]], probability: list[float],
                               high_conf: list[bool], draws: int = 1000) -> float | None:
    by_date: dict[str, list[float]] = defaultdict(list)
    for row, prob, selected in zip(rows, probability, high_conf):
        if selected:
            direction = 1.0 if prob >= 0.5 else -1.0
            by_date[row["event_date"]].append(direction * row["fwd_ret_30m"] - COST)
    dates = sorted(by_date)
    if len(dates) < 2:
        return None
    rng = random.Random(22101)
    samples: list[float] = []
    for _ in range(draws):
        chosen = [dates[rng.randrange(len(dates))] for _ in dates]
        values = [value for date in chosen for value in by_date[date]]
        samples.append(sum(values) / len(values))
    return sorted(samples)[int(0.025 * (draws - 1))]


def evaluate_hooks(rows: list[dict[str, Any]], predictions: dict[str, dict[str, str]],
                   available_hooks: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row["event_time_utc"], row["event_id"]))
    baseline_records = [predictions[row["event_id"]] for row in ordered]
    confidence = [float(record["V69_confidence_signal"]) for record in baseline_records]
    high_conf = [str(record["V69_high_conf"]).strip().lower() in {"1", "true", "yes"}
                 for record in baseline_records]
    report: dict[str, Any] = {}
    for name in COMPONENT_HOOKS:
        if name not in set(available_hooks):
            report[name] = {"status": "NOT_EVALUATED_MISSING_FROZEN_SCORER_BUNDLE",
                            "required": COMPONENT_HOOKS[name]["required"]}
            continue
        column = COMPONENT_HOOKS[name]["prediction_column"]
        probability = [float(record[column]) for record in baseline_records]
        overall = summarize(ordered, probability, confidence, high_conf)
        overall["date_block_bootstrap_highconf_net_lower95"] = date_block_bootstrap_lower(
            ordered, probability, high_conf
        )
        by_source = {}
        for source in sorted(ALLOWED_SOURCE_FAMILIES):
            indices = [index for index, row in enumerate(ordered) if row["source_family"] == source]
            source_rows = [ordered[index] for index in indices]
            by_source[source] = summarize(
                source_rows, [probability[index] for index in indices],
                [confidence[index] for index in indices], [high_conf[index] for index in indices],
            )
        report[name] = {
            "status": "ZERO_TUNE_EVALUATED_NO_SELECTION",
            "primary_axes": COMPONENT_HOOKS[name]["primary_axes"],
            "overall": overall, "by_source_family": by_source,
            "confidence_and_high_conf_source": "V69_LOCKED_CHAMPION",
            "fit_tune_threshold_or_polarity_change": False,
        }
    require(report["V69_LOCKED_CHAMPION"]["status"] == "ZERO_TUNE_EVALUATED_NO_SELECTION",
            "locked V69 zero-tune was not evaluated")
    return report


def polarity_hook_status(observed_sources: set[str]) -> dict[str, Any]:
    result = {}
    for name, hook in OLD_POLARITY_HOOKS.items():
        scope = set(hook.get("source_scope", observed_sources))
        supported = sorted(scope & observed_sources)
        result[name] = {
            "status": "HOOK_DEFINED_FROZEN_EVIDENCE_REQUIRED" if supported else "UNTESTABLE_SOURCE_ABSENT",
            "supported_sources": supported,
            "missing_sources": sorted(scope - observed_sources),
            "roles": hook["roles"],
            "selection_or_tuning_allowed": False,
            "outer_result_may_change_frozen_policy": False,
        }
    return result


def run_evaluation(manifest_path: Path) -> dict[str, Any]:
    state, activation = require_ready_epoch()
    authority = verify_frozen_authority()
    rows, extension_audit = load_extension(activation)
    predictions, bundle_audit = load_prediction_bundle(manifest_path, rows, activation)
    component_report = evaluate_hooks(rows, predictions, bundle_audit["available_hooks"])
    return {
        "version": VERSION, "hypothesis": HYPOTHESIS,
        "status": "ZERO_TUNE_TRANSFER_EVALUATION_ONLY",
        "registered": False, "production_intervention_authorized": False,
        "seal_authorized": False, "final_meta_authorized": False,
        "epoch": {"status": state["status"], "data_epoch_sha256": activation["data_epoch_sha256"]},
        "extension_audit": extension_audit, "authority": authority,
        "prediction_bundle": bundle_audit, "component_zero_tune": component_report,
        "old_polarity_replication_hooks": polarity_hook_status(set(extension_audit["sources"])),
        "no_retraining": True, "no_threshold_change": True, "no_polarity_change": True,
        "no_outer_driven_selection": True, "reserved_role_labels_opened": False,
    }


def scaffold_audit() -> dict[str, Any]:
    state = inspect_epoch_state()
    authority = verify_frozen_authority()
    return {
        "version": VERSION, "hypothesis": HYPOTHESIS,
        "status": "PREPARED_NOT_REGISTERED_EPOCH_BLOCKED" if state["status"] not in READY_STATES else "PREPARED_NOT_REGISTERED_EPOCH_READY",
        "registered": False, "full_experiment_executed": False,
        "output_or_staging_written": False, "dev_extension_labels_read": False,
        "seal_or_final_labels_opened": False, "epoch_state": state,
        "authority": authority, "component_hooks": COMPONENT_HOOKS,
        "old_polarity_hooks": polarity_hook_status(ALLOWED_SOURCE_FAMILIES),
        "zero_tune_bundle_required": True,
        "missing_frozen_bundle_behavior": "FAIL_CLOSED_NO_RETRAIN",
    }


def synthetic_smoke() -> dict[str, Any]:
    rows = [
        {"event_id": f"S{i}", "event_group_id": f"G{i}", "source_family": "US_SEC" if i < 4 else "KR_NEWS",
         "market": "US" if i < 4 else "KR", "ticker": str(i),
         "event_time_utc": f"2026-01-0{i + 1}T00:00:00+00:00", "event_date": f"2026-01-0{i + 1}",
         "y": i % 2, "fwd_ret_30m": 0.01 if i % 2 else -0.01}
        for i in range(8)
    ]
    probability = [0.1, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6]
    confidence = [abs(value - 0.5) * 2.0 for value in probability]
    metrics = summarize(rows, probability, confidence, [index % 2 == 1 for index in range(8)])
    require(metrics["auc"] == 1.0 and metrics["balanced_accuracy"] == 1.0,
            "synthetic direction metric hook failed")
    return {"status": "PASS_NO_WRITE_NO_REAL_LABEL_READ", "metrics_hook": metrics,
            "component_hook_count": len(COMPONENT_HOOKS),
            "old_polarity_hook_count": len(OLD_POLARITY_HOOKS)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("readiness", "scaffold-audit", "synthetic-smoke", "evaluate"),
                       default="readiness")
    value.add_argument("--prediction-manifest", type=Path)
    return value


def emit(payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["resource_contract"] = configure_bounded_resources()
    payload["emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def main() -> None:
    args = parser().parse_args()
    resources = configure_bounded_resources()
    try:
        if args.mode == "scaffold-audit":
            payload = scaffold_audit()
        elif args.mode == "synthetic-smoke":
            payload = synthetic_smoke()
        elif args.mode == "evaluate":
            # The live epoch gate is the first authority check even when the
            # caller omitted the later frozen-bundle argument.  This keeps the
            # observable fail-closed order identical to the draft contract and
            # proves that no evaluation dependency is consulted while the
            # extension epoch is still blocked.
            state = inspect_epoch_state()
            if state["status"] not in READY_STATES:
                raise EpochNotReady(
                    f"CURRENT_DATA_EPOCH is {state['status']!r}; V221 evaluation is blocked"
                )
            require(args.prediction_manifest is not None, "--prediction-manifest is required for evaluate")
            payload = run_evaluation(args.prediction_manifest)
        else:
            state = inspect_epoch_state()
            if state["status"] not in READY_STATES:
                raise EpochNotReady(
                    f"CURRENT_DATA_EPOCH is {state['status']!r}; V221 evaluation is blocked"
                )
            payload = {"status": "READY_EVALUATION_REQUIRES_FROZEN_PREDICTION_BUNDLE",
                       "epoch_state": state}
        payload["resource_contract"] = resources
        payload["emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    except EpochNotReady as error:
        print(json.dumps({
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": "FAIL_CLOSED_EPOCH_NOT_READY",
            "error": str(error), "dev_extension_labels_read": False,
            "seal_or_final_labels_opened": False, "output_written": False,
            "resource_contract": resources,
        }, ensure_ascii=False, sort_keys=True, indent=2))
        raise SystemExit(20) from None
    except ContractError as error:
        print(json.dumps({
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": "FAIL_CLOSED_CONTRACT_ERROR",
            "error": str(error), "seal_or_final_labels_opened": False,
            "output_written": False, "resource_contract": resources,
        }, ensure_ascii=False, sort_keys=True, indent=2))
        raise SystemExit(21) from None


if __name__ == "__main__":
    main()
