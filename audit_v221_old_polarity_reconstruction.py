"""Read-only reconstruction of authoritative V64-V69 OLD DEV polarity evidence.

The audit is intentionally independent of the V64-V69 runners.  It uses only
the Python standard library, pins itself to the bounded preparation contract
(logical CPUs 30-31, two numeric threads, GPU hidden), verifies every file in
each committed artifact manifest, and then reconstructs the claims that the
committed reports actually support.  It never writes an output or imports a
model/acquisition/provider module.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEFAULT_EXPECTED = ROOT / "research" / "V221_OLD_POLARITY_RECONSTRUCTION.json"
CPU_IDS = (30, 31)
CPU_MASK = sum(1 << item for item in CPU_IDS)
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
SOURCES = ("US_SEC", "US_NEWS", "KR_NEWS", "KR_KIND")
BLOCKS = (
    "return_path",
    "volatility_range",
    "volume_activity",
    "trend_position",
    "market_relative",
    "event_clock",
    "event_lexical_flags",
    "price_scale",
    "categorical_context",
    "text_content",
)
DATA_EPOCH_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"

PINNED_COMMITS: dict[int, dict[str, Any]] = {
    64: {
        "experiment_id": "fd5b219da068f98ef0165558606e1d44341cc7d86c57b45a2a868e851bc8db83",
        "manifest_sha256": "ce7c719bcceaf6f0ecb553c07cc45f9224209dca170d24b464eeb46201370dd1",
        "run_status": "DIAGNOSTIC_ONLY",
    },
    65: {
        "experiment_id": "544646f5deded3a59690df71a6a21d0ee5f2771636350fdd1ca8c1401cdc74b7",
        "manifest_sha256": "f6386e57b29d2e37d31d008e7891827380bbcb7db36abbf96cad9f8d7c472578",
        "run_status": "MATERIAL_EXPERIMENT_FAIL_NOOP",
    },
    66: {
        "experiment_id": "4a36b05800e6f9ab300e0e9d8ccd6489ba2f41c8d7ea858a6006a526c351ed5d",
        "manifest_sha256": "74198a5011a587d7ba2d60ecbffaac7aa78a33386ec3da3beb49a1f40293986f",
        "run_status": "MATERIAL_EXPERIMENT_FAIL_NOOP",
    },
    67: {
        "experiment_id": "fe5145796f202870a2b4272e20487d732e99dfc7c333e9e2fbf34788acbb00c5",
        "manifest_sha256": "8273a7cb933471bc1e2228ee7dc55b8de4eaaff01e4e5a193e21dc55f6fae865",
        "run_status": "DIAGNOSTIC_ONLY",
    },
    68: {
        "experiment_id": "a76f8f082c6dbe5d62a03d81a6d3d4f8902706f009fe39f4bbae73e720ad5d13",
        "manifest_sha256": "fcaa9472313e0a4ae6765e37cbd55ce0f3f888e3825935beabbf85fa4ee0eea8",
        "run_status": "MATERIAL_EXPERIMENT_FAIL_NOOP",
    },
    69: {
        "experiment_id": "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740",
        "manifest_sha256": "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836",
        "run_status": "MATERIAL_PASS",
    },
}

PINNED_RUNNERS = {
    "experiment_v64_feature_polarity_diagnostic.py": "8dbdf7911ba90e9848dc78eed9a821fb9b9b8e245c73d88e9e70721b3be50285",
    "experiment_v65_us_news_confidence_reversal.py": "c0ebe297678a128dd83d4883bfc227d31693413ebf74a3c5ab978b993479938d",
    "experiment_v66_extreme_reversal_guard.py": "067006451e06580658195b64bae6a6362587b06a484d7a5803346b1aef40f64d",
    "experiment_v67_direction_opportunity_polarity_diagnostic.py": "fcdfe200a482c2813fbae46fa7462874e4b94ee472823e76f078c544014052f4",
    "experiment_v68_kr_news_direction_opportunity_router.py": "17c2ce54a911ae997e7ed779480842be222f8e8d28193680300f0baf142a97e6",
    "experiment_v69_us_sec_structured_agreement_selector.py": "86be2705c7c1b18241605350122d86483b10e4a7aa875dd8d629a78c72fc7136",
}

EVIDENCE_PATHS = (
    "output_V64/CONFIDENCE_RELIABILITY_REPORT.json",
    "output_V64/EXTREME_REVERSAL_REPORT.json",
    "output_V64/POLARITY_STABILITY_REPORT.csv",
    "output_V65/US_NEWS_CONFIDENCE_REVERSAL_REPORT.json",
    "output_V66/EXTREME_REVERSAL_GUARD_REPORT.json",
    "output_V67/DIRECTION_OPPORTUNITY_POLARITY_MAP_V67.json",
    "output_V67/POLARITY_STABILITY_REPORT.csv",
    "output_V68/KR_NEWS_DIRECTION_OPPORTUNITY_ROUTER_REPORT.json",
    "output_V69/US_SEC_STRUCTURED_AGREEMENT_SELECTOR_REPORT.json",
    "output_V69/AGREEMENT_COHORT_AUDIT.json",
    "output_V69/DEV_ROBUSTNESS_REPORT.json",
    "output_V69/ECONOMIC_ROBUSTNESS_REPORT.json",
    "output_V69/DIRECTION_IMMUTABILITY_AUDIT.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    require(str(value).strip().lower() in {"true", "false"}, f"invalid boolean: {value!r}")
    return str(value).strip().lower() == "true"


def maybe_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def bounded_runtime() -> dict[str, Any]:
    for name in THREAD_ENV_NAMES:
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
        kernel32.GetProcessAffinityMask.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        kernel32.GetProcessAffinityMask.restype = ctypes.c_bool
        process = kernel32.GetCurrentProcess()
        require(
            bool(kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(CPU_MASK))),
            "failed to pin OLD polarity audit to logical CPUs 30-31",
        )
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        require(
            bool(kernel32.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask))),
            "failed to query OLD polarity audit affinity",
        )
        active = [item for item in range(64) if process_mask.value & (1 << item)]
        mask_hex = f"0x{process_mask.value:08X}"
    else:
        require(hasattr(os, "sched_setaffinity"), "strict CPU affinity is unsupported")
        os.sched_setaffinity(0, set(CPU_IDS))
        active = sorted(os.sched_getaffinity(0))
        mask_hex = hex(sum(1 << item for item in active))
    require(active == list(CPU_IDS), f"bounded CPU affinity drifted: {active}")
    require(all(os.environ[name] == "2" for name in THREAD_ENV_NAMES), "numeric thread limit drifted")
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "-1", "GPU is not hidden")
    return {
        "logical_cpu_ids": active,
        "affinity_mask_hex": mask_hex,
        "numeric_threads": 2,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "nvidia_visible_devices": os.environ["NVIDIA_VISIBLE_DEVICES"],
        "gpu_hidden": True,
        "gpu_model_calls": 0,
    }


def verify_committed_outputs() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for version, pinned in PINNED_COMMITS.items():
        directory = ROOT / f"output_V{version}"
        commit = load_json(f"output_V{version}/COMMIT.json")
        status = load_json(f"output_V{version}/RUN_STATUS.json")
        manifest_path = directory / "ARTIFACT_MANIFEST.json"
        manifest_hash = sha256(manifest_path)
        manifest = load_json(f"output_V{version}/ARTIFACT_MANIFEST.json")
        require(commit["version"] == version, f"V{version} COMMIT version mismatch")
        require(status["version"] == version, f"V{version} RUN_STATUS version mismatch")
        require(commit["data_sha"] == DATA_EPOCH_SHA256, f"V{version} data epoch changed")
        require(commit["experiment_id"] == pinned["experiment_id"], f"V{version} experiment id changed")
        require(manifest["experiment_id"] == pinned["experiment_id"], f"V{version} manifest experiment id changed")
        require(commit["manifest_sha256"] == pinned["manifest_sha256"], f"V{version} pinned manifest changed")
        require(manifest_hash == pinned["manifest_sha256"], f"V{version} manifest bytes changed")
        require(status["status"] == pinned["run_status"], f"V{version} run status changed")
        require(status["phase"] == "RESEARCH_FAIL", f"V{version} phase contract changed")
        require(status["seal_state"] == "UNOPENED", f"V{version} seal state changed")
        require(status["final_status"] == "CONTINUE", f"V{version} final status changed")
        byte_total = 0
        for name, item in sorted(manifest["files"].items()):
            path = directory / name
            require(path.is_file(), f"V{version} manifested file missing: {name}")
            actual_size = path.stat().st_size
            actual_hash = sha256(path)
            require(actual_size == int(item["bytes"]), f"V{version} byte size changed: {name}")
            require(actual_hash == item["sha256"], f"V{version} artifact hash changed: {name}")
            byte_total += actual_size
        versions[str(version)] = {
            "experiment_id": commit["experiment_id"],
            "manifest_sha256": manifest_hash,
            "manifested_file_count": len(manifest["files"]),
            "manifested_bytes": byte_total,
            "run_status": status["status"],
            "phase": status["phase"],
            "seal_state": status["seal_state"],
            "final_status": status["final_status"],
            "all_manifest_members_verified": True,
        }
    runners = {}
    for name, expected in PINNED_RUNNERS.items():
        actual = sha256(ROOT / name)
        require(actual == expected, f"pinned runner changed: {name}")
        runners[name] = actual
    evidence_files = {relative: sha256(ROOT / relative) for relative in EVIDENCE_PATHS}
    return {
        "data_epoch_sha256": DATA_EPOCH_SHA256,
        "versions": versions,
        "runner_sha256": runners,
        "evidence_file_sha256": evidence_files,
        "all_commits_and_manifest_members_verified": True,
    }


def confidence_measure(report: dict[str, Any]) -> dict[str, Any]:
    top = report["high_confidence_top_slices"]["top_20_percent"]
    bottom = report["bottom_20_percent"]
    frozen_high = report["frozen_v58_high_confidence"]
    frozen_other = report["frozen_v58_non_high_confidence"]
    return {
        "n": report["n"],
        "confidence_correctness_auc": report["confidence_correctness_auc"],
        "top20_n": top["n"],
        "top20_accuracy": top["accuracy"],
        "bottom20_n": bottom["n"],
        "bottom20_accuracy": bottom["accuracy"],
        "top20_minus_bottom20_accuracy": top["accuracy"] - bottom["accuracy"],
        "accuracy_spearman_by_decile": report["monotonicity"]["accuracy_spearman_by_bin"],
        "frozen_highconf_n": frozen_high["n"],
        "frozen_highconf_accuracy": frozen_high["accuracy"],
        "frozen_non_highconf_n": frozen_other["n"],
        "frozen_non_highconf_accuracy": frozen_other["accuracy"],
        "reversal_checks": report["reversal_checks"],
    }


def reconstruct_confidence() -> dict[str, Any]:
    report = load_json("output_V64/CONFIDENCE_RELIABILITY_REPORT.json")
    assessments = {
        "OVERALL": "MIXED_BY_CONFIDENCE_DEFINITION",
        "US_SEC": "NO_REVERSAL_WEAK_POSITIVE_TENDENCY",
        "US_NEWS": "HIGH_CONFIDENCE_REVERSAL",
        "KR_NEWS": "NEAR_NULL_MIXED_NEGATIVE_TENDENCY",
        "KR_KIND": "SMALL_N_NO_NEGATIVE_AUC_EVIDENCE",
    }
    scopes = {}
    for scope in ("OVERALL", *SOURCES):
        scopes[scope] = {
            "assessment": assessments[scope],
            "confidence_role_block_polarity": "NOT_MEASURED_IN_V64_V69",
            "frozen_v58_confidence": confidence_measure(report["scopes"][scope]["frozen_v58_confidence"]),
            "model_probability_margin": confidence_measure(report["scopes"][scope]["model_probability_margin"]),
        }
    us_news = scopes["US_NEWS"]["frozen_v58_confidence"]
    require(abs(us_news["confidence_correctness_auc"] - 0.4613656504785537) < 1e-15, "US_NEWS CCA changed")
    require(abs(us_news["top20_accuracy"] - 0.46596858638743455) < 1e-15, "US_NEWS top20 changed")
    require(abs(us_news["bottom20_accuracy"] - 0.581151832460733) < 1e-15, "US_NEWS bottom20 changed")
    require(us_news["reversal_checks"]["high_confidence_reversal"] is True, "US_NEWS reversal flag changed")
    return {
        "authority": "V64 frozen V58 chronological OOF confidence diagnostic on OLD DEV",
        "scopes": scopes,
        "us_news_high_confidence_reversal": {
            "evidenced": True,
            "confidence_correctness_auc": us_news["confidence_correctness_auc"],
            "top20_accuracy": us_news["top20_accuracy"],
            "bottom20_accuracy": us_news["bottom20_accuracy"],
            "top20_minus_bottom20_accuracy": us_news["top20_minus_bottom20_accuracy"],
            "accuracy_spearman_by_decile": us_news["accuracy_spearman_by_decile"],
            "frozen_highconf_accuracy": us_news["frozen_highconf_accuracy"],
            "frozen_non_highconf_accuracy": us_news["frozen_non_highconf_accuracy"],
            "qualification": "OLD DEV source-level reliability evidence; not a feature-block cause and not NEW DATA replication",
        },
        "implementation_caveat": (
            "V64 high_confidence_reversal is any of four checks, not their conjunction. "
            "US_NEWS is unusually strong because correctness AUC, top-vs-bottom accuracy, "
            "frozen-HC-vs-other accuracy, and decile correlation all point negatively."
        ),
    }


def reconstruct_v64_polarity() -> list[dict[str, Any]]:
    rows = load_csv("output_V64/POLARITY_STABILITY_REPORT.csv")
    result = []
    for source in ("US_NEWS", "US_SEC"):
        for block in BLOCKS:
            row = next(item for item in rows if item["source_family"] == source and item["block"] == block)
            result.append({
                "source_family": source,
                "feature_block": block,
                "polarity": row["polarity"],
                "fold_n": int(row["fold_n"]),
                "stability": float(row["stability_to_median_sign"]),
                "median_combined_contribution": float(row["median_combined_contribution"]),
                "fallback_reason": row["fallback_reason"] or None,
                "fold_sign_counts": {
                    "+": int(row["positive_fold_n"]),
                    "0": int(row["zero_fold_n"]),
                    "-": int(row["negative_fold_n"]),
                },
            })
    return result


def reconstruct_v67_roles() -> list[dict[str, Any]]:
    rows = load_csv("output_V67/POLARITY_STABILITY_REPORT.csv")
    result = []
    for source in SOURCES:
        for block in BLOCKS:
            row = next(item for item in rows if item["source_family"] == source and item["feature_block"] == block)
            result.append({
                "source_family": source,
                "feature_block": block,
                "source_n": int(row["source_n"]),
                "source_supported": as_bool(row["source_supported"]),
                "direction": {
                    "polarity": row["direction_polarity"],
                    "stability": float(row["direction_stability"]),
                    "median_late_score_delta": maybe_float(row["direction_median_delta"]),
                    "fallback_reason": row["direction_fallback_reason"] or None,
                },
                "confidence": {
                    "polarity": None,
                    "status": "NOT_MEASURED_AT_BLOCK_LEVEL_IN_V64_V69",
                },
                "opportunity": {
                    "polarity": row["opportunity_polarity"],
                    "stability": float(row["opportunity_stability"]),
                    "median_late_score_delta": maybe_float(row["opportunity_median_delta"]),
                    "fallback_reason": row["opportunity_fallback_reason"] or None,
                },
                "strict_direction_opportunity_opposite": as_bool(row["direction_opportunity_reversal"]),
            })
    return result


def source_role_summary(rows: list[dict[str, Any]], confidence: dict[str, Any]) -> dict[str, Any]:
    summaries = {}
    for source in SOURCES:
        selected = [row for row in rows if row["source_family"] == source]
        direction = {
            row["feature_block"]: {
                "polarity": row["direction"]["polarity"],
                "stability": row["direction"]["stability"],
            }
            for row in selected if row["direction"]["polarity"] != "0"
        }
        opportunity = {
            row["feature_block"]: {
                "polarity": row["opportunity"]["polarity"],
                "stability": row["opportunity"]["stability"],
            }
            for row in selected if row["opportunity"]["polarity"] != "0"
        }
        mismatches = [
            {
                "feature_block": row["feature_block"],
                "direction": row["direction"]["polarity"],
                "opportunity": row["opportunity"]["polarity"],
                "strict_opposite": row["strict_direction_opportunity_opposite"],
            }
            for row in selected
            if row["direction"]["polarity"] != row["opportunity"]["polarity"]
        ]
        summaries[source] = {
            "v67_source_n": selected[0]["source_n"],
            "v67_source_supported": selected[0]["source_supported"],
            "direction_nonzero_blocks": direction,
            "direction_zero_blocks": [
                row["feature_block"] for row in selected if row["direction"]["polarity"] == "0"
            ],
            "confidence_source_level_assessment": confidence["scopes"][source]["assessment"],
            "confidence_block_polarity": "NOT_MEASURED_IN_V64_V69",
            "opportunity_nonzero_blocks": opportunity,
            "opportunity_zero_blocks": [
                row["feature_block"] for row in selected if row["opportunity"]["polarity"] == "0"
            ],
            "direction_opportunity_role_mismatches": mismatches,
        }
    return summaries


def contradictions(v64: list[dict[str, Any]], v67: list[dict[str, Any]]) -> dict[str, Any]:
    changes = []
    opposite = []
    stable_matches = []
    for old in v64:
        new = next(
            row for row in v67
            if row["source_family"] == old["source_family"] and row["feature_block"] == old["feature_block"]
        )
        new_sign = new["direction"]["polarity"]
        if old["polarity"] != new_sign:
            item = {
                "source_family": old["source_family"],
                "feature_block": old["feature_block"],
                "v64_direction_diagnostic": {
                    "polarity": old["polarity"],
                    "stability": old["stability"],
                    "median_combined_contribution": old["median_combined_contribution"],
                },
                "v67_direction_diagnostic": new["direction"],
            }
            changes.append(item)
            if old["polarity"] in {"+", "-"} and new_sign in {"+", "-"}:
                opposite.append(item)
        elif old["polarity"] in {"+", "-"}:
            stable_matches.append({
                "source_family": old["source_family"],
                "feature_block": old["feature_block"],
                "polarity": old["polarity"],
                "v64_stability": old["stability"],
                "v67_stability": new["direction"]["stability"],
            })
    premises = []
    for block in ("categorical_context", "volatility_range"):
        old = next(row for row in v64 if row["source_family"] == "US_NEWS" and row["feature_block"] == block)
        new = next(row for row in v67 if row["source_family"] == "US_NEWS" and row["feature_block"] == block)
        premises.append({
            "feature_block": block,
            "v64_polarity": old["polarity"],
            "v64_stability": old["stability"],
            "v67_direction_polarity": new["direction"]["polarity"],
            "v67_direction_stability": new["direction"]["stability"],
            "v67_opportunity_polarity": new["opportunity"]["polarity"],
            "v67_opportunity_stability": new["opportunity"]["stability"],
        })
    require(len(changes) == 8, f"V64/V67 direction classification-change count drifted: {len(changes)}")
    require(len(opposite) == 0, "unexpected direct nonzero V64/V67 direction sign opposition")
    return {
        "classification_change_count": len(changes),
        "classification_changes": changes,
        "direct_nonzero_sign_opposition_count": len(opposite),
        "direct_nonzero_sign_oppositions": opposite,
        "same_nonzero_sign_count": len(stable_matches),
        "same_nonzero_signs": stable_matches,
        "v65_v66_negative_block_premise_conflicts_with_later_v67": premises,
        "interpretation": (
            "These are method-dependent diagnostic contradictions, mainly stable-sign versus conservative zero, "
            "not direct + versus - direction flips. V64 mixed 50% full-minus-ablated late-score delta with "
            "50% block-only late-score minus 0.5, used a 0.001 epsilon and a 0.46-0.54 cutoff grid. "
            "V67 used full-minus-ablated only, a different direction score including edge, a 0.0015 epsilon, "
            "and a 0.48-0.52 cutoff grid. The outputs must not be merged into one authoritative sign."
        ),
    }


def compact_confidence_metrics(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": item["n"],
        "highconf_n": item["highconf_n"],
        "highconf_coverage": item["highconf_coverage"],
        "highconf_accuracy": item["highconf_accuracy"],
        "highconf_mean_signed_net": item["highconf_mean_signed_net"],
        "confidence_correctness_auc": item["confidence_correctness_auc"],
    }


def reconstruct_interventions() -> dict[str, Any]:
    v65 = load_json("output_V65/US_NEWS_CONFIDENCE_REVERSAL_REPORT.json")
    e65 = v65["evaluation"]
    v66 = load_json("output_V66/EXTREME_REVERSAL_GUARD_REPORT.json")
    e66 = v66["evaluation"]
    v68 = load_json("output_V68/KR_NEWS_DIRECTION_OPPORTUNITY_ROUTER_REPORT.json")
    e68 = v68["evaluation"]
    v69 = load_json("output_V69/US_SEC_STRUCTURED_AGREEMENT_SELECTOR_REPORT.json")
    e69 = v69["evaluation"]
    dev69 = load_json("output_V69/DEV_ROBUSTNESS_REPORT.json")
    cohorts = load_json("output_V69/AGREEMENT_COHORT_AUDIT.json")
    direction = load_json("output_V69/DIRECTION_IMMUTABILITY_AUDIT.json")
    require(v65["status"] == "MATERIAL_EXPERIMENT_FAIL_NOOP" and not e65["material_pass"], "V65 status drifted")
    require(v66["status"] == "MATERIAL_EXPERIMENT_FAIL_NOOP" and not e66["material_pass"], "V66 status drifted")
    require(v68["status"] == "MATERIAL_EXPERIMENT_FAIL_NOOP" and not e68["material_pass"], "V68 status drifted")
    require(v69["status"] == "MATERIAL_PASS" and e69["material_pass"], "V69 material result drifted")
    require(direction["exact_unchanged"] and direction["direction_mismatch_n"] == 0, "V69 direction changed")
    baseline65 = e65["candidate_metrics"]["V58_FROZEN_CONFIDENCE"]
    baseline66 = e66["candidate_metrics"]["V58_FROZEN_CONFIDENCE_NOOP"]
    return {
        "V65": {
            "scope": "US_NEWS",
            "status": v65["status"],
            "premise_from_v64": v65["input_audit"]["v64_stable_negative_blocks"],
            "method": "past-fit P(correct) meta candidates plus late-expert zero/penalty/inverse confidence candidates",
            "eligible_oof_n": e65["eligible_oof_n"],
            "baseline_inner_threshold": compact_confidence_metrics(baseline65),
            "inner_selected_challenger": compact_confidence_metrics(e65["inner_selected_metrics"]),
            "material_checks": e65["material_checks"],
            "fallback": v65["fallback"],
            "authoritative_result": "FAILED_CLOSED_TO_EXACT_V58_CONFIDENCE_AND_HIGH_CONF",
        },
        "V66": {
            "scope": "US_NEWS",
            "status": v66["status"],
            "method": "sparse extreme-tail positive/absolute contribution penalty, clip, or saturation guard",
            "eligible_oof_n": e66["eligible_oof_n"],
            "baseline_inner_threshold": compact_confidence_metrics(baseline66),
            "inner_selected_challenger": compact_confidence_metrics(e66["inner_selected_metrics"]),
            "guard_selected_fold_fraction": e66["guard_selected_fold_fraction"],
            "material_checks": e66["material_checks"],
            "fallback": v66["fallback"],
            "authoritative_result": "FAILED_CLOSED_TO_EXACT_V58_CONFIDENCE_AND_HIGH_CONF",
        },
        "V68": {
            "scope": "KR_NEWS_ONLY",
            "status": v68["status"],
            "v67_strict_opposite_blocks_tested": v68["v67_evidence"],
            "eligible_outer_oof_n": e68["eligible_outer_oof_n"],
            "baseline_direction": e68["baseline_direction"],
            "candidate_direction": e68["candidate_direction"],
            "baseline_opportunity": e68["baseline_opportunity"],
            "candidate_opportunity": e68["candidate_opportunity"],
            "direction_positive_fold_fraction": e68["direction_positive_fold_fraction"],
            "opportunity_positive_fold_fraction": e68["opportunity_positive_fold_fraction"],
            "material_checks": e68["material_checks"],
            "fallback": v68["fallback"],
            "authoritative_result": "V67_ROLE_REVERSALS_DID_NOT_TRANSFER_TO_THE_NESTED_ROUTER",
        },
        "V69": {
            "scope": "US_SEC_CONFIDENCE_RANKING_ONLY",
            "status": v69["status"],
            "candidate_contract": v69["candidate_contract"],
            "outer_evidence_rows": e69["outer_evidence_rows"],
            "outer_evidence_fraction_of_us_sec": e69["outer_evidence_fraction_of_us_sec"],
            "outer_baseline": e69["outer_baseline"],
            "outer_candidate": e69["outer_candidate"],
            "outer_delta": e69["outer_delta"],
            "outer_fold_net_deltas": e69["outer_fold_net_deltas"],
            "positive_fold_fraction": e69["positive_fold_fraction"],
            "nonnegative_fold_fraction": e69["nonnegative_fold_fraction"],
            "full_us_sec_highconf_net": e69["full_us_sec_hc_net"],
            "bootstrap": e69["bootstrap"],
            "agreement_cohorts": cohorts,
            "direction_immutability": direction,
            "material_checks": e69["material_checks"],
            "canonical_gate_qualification": {
                "passed": dev69["selected"]["research_gate"]["passed"],
                "total": dev69["selected"]["research_gate"]["total"],
                "robust_survivor": dev69["selected"]["research_gate"]["robust_survivor"],
                "selected_highconf_net_lower95": dev69["selected"]["robustness"]["bootstrap"][
                    "highconf_strategy_net_lower95"
                ],
                "run_phase": "RESEARCH_FAIL",
                "seal_state": "UNOPENED",
                "seal_authorized": False,
                "meaning": "material selector pass, not canonical 14/14 robust-survivor or seal pass",
            },
            "authoritative_effect": (
                "Agreement re-ranked a fixed-size US_SEC HC slice and improved OLD DEV HC net. "
                "Outer HC accuracy was unchanged and confidence-correctness AUC decreased within its guard."
            ),
        },
    }


def build_reconstruction() -> dict[str, Any]:
    integrity = verify_committed_outputs()
    confidence = reconstruct_confidence()
    v64 = reconstruct_v64_polarity()
    v67 = reconstruct_v67_roles()
    mismatch_rows = [
        row for row in v67
        if row["direction"]["polarity"] != row["opportunity"]["polarity"]
    ]
    strict_rows = [row for row in v67 if row["strict_direction_opportunity_opposite"]]
    require(len(mismatch_rows) == 16, "V67 role-mismatch count drifted")
    require(len(strict_rows) == 2, "V67 strict role-opposition count drifted")
    return {
        "schema_version": 1,
        "title": "V221 OLD DEV polarity/confidence reconstruction from committed V64-V69 evidence",
        "scope": {
            "data_epoch": "V36-era OLD DEV only",
            "data_epoch_sha256": DATA_EPOCH_SHA256,
            "versions": [64, 65, 66, 67, 68, 69],
            "new_data_used": False,
            "research_seal_or_final_labels_used": False,
            "production_intervention_authorized_by_this_audit": False,
        },
        "integrity": integrity,
        "method_authority": {
            "V64": (
                "Direction-target linear probe; per-fold polarity is 50% (full late score - ablated late score) "
                "+ 50% (block-only late score - 0.5); nonzero requires >=0.70 fold-sign stability."
            ),
            "V67": (
                "Separate direction and opportunity linear probes; per-role polarity is full late score minus "
                "leave-one-block-out late score; nonzero requires >=0.70 fold-sign stability; unsupported KR_KIND is zero."
            ),
            "shared_constraints": (
                "Chronological outer folds, inner-past preprocessing/selection, strict 35-minute embargo, "
                "no raw feature sign flip, diagnostic ablation proxy rather than causal SHAP."
            ),
        },
        "confidence_reliability": confidence,
        "v64_direction_diagnostic_rows": v64,
        "v67_role_polarity_rows": v67,
        "source_role_summary": source_role_summary(v67, confidence),
        "role_divergence": {
            "direction_not_equal_opportunity_count_including_zero_vs_nonzero": len(mismatch_rows),
            "strict_nonzero_opposite_count": len(strict_rows),
            "strict_nonzero_opposites": [
                {
                    "source_family": row["source_family"],
                    "feature_block": row["feature_block"],
                    "direction": row["direction"],
                    "opportunity": row["opportunity"],
                }
                for row in strict_rows
            ],
        },
        "v64_v67_contradictions": contradictions(v64, v67),
        "intervention_results": reconstruct_interventions(),
        "evidence_boundary": {
            "genuinely_evidenced": [
                "US_NEWS OLD DEV frozen-confidence reversal: CCA 0.4613656504785537, top20 accuracy 0.46596858638743455, bottom20 accuracy 0.581151832460733, decile-accuracy Spearman -0.6484848484848483.",
                "V64 emitted diagnostic stable block signs for US_NEWS and US_SEC, but its sign statistic is method-specific.",
                "V67 emitted separate diagnostic direction/opportunity block signs for supported US_SEC, US_NEWS, and KR_NEWS; KR_KIND was forced to zero for insufficient support.",
                "V67 found two strict opposite nonzero role signs: KR_NEWS event_clock direction-/opportunity+ and text_content direction-/opportunity+.",
                "V65 and V66 US_NEWS confidence-reversal interventions failed their material gates and fell back exactly to V58.",
                "V68 KR_NEWS polarity router failed; direction, opportunity, HC accuracy, and HC net did not improve sufficiently and the output fell back exactly to V58.",
                "V69 US_SEC agreement ranking materially improved HC economic net while preserving direction probability, direction, HC count, and aggregate outer HC accuracy.",
                "V69 did not improve correctness ranking: outer confidence-correctness AUC moved by -0.0018659572591023244.",
            ],
            "not_evidenced_or_only_hypothesized": [
                "No V64-V69 result is a contribution-level z_full - z_without_B counterfactual with alpha in {-1,0,+0.5,+1,+1.5}.",
                "No V64-V69 result proves that contribution inversion (-B) is better than retention (+B) or ablation (0B).",
                "No source-by-feature-block confidence polarity map was measured; only source-level confidence reliability and targeted guard candidates exist.",
                "The V64 US_NEWS negative blocks are not stable authority across methods: V67 maps both categorical_context and volatility_range direction to zero, and volatility_range opportunity to positive.",
                "V69 agreement does not prove agreement predicts direction correctness; in its committed cohorts AGREE accuracy is lower than DISAGREE accuracy, while AGREE mean net is less negative.",
                "V69 agreement is not evidence for US_NEWS reversal repair; its scope is US_SEC confidence ranking.",
                "No OLD polarity claim has NEW Massive/KIS replication in V64-V69.",
                "No V64-V69 polarity result authorizes a production sign flip, seal opening, or final-meta conclusion.",
            ],
            "authoritative_research_conclusion": (
                "REVERSAL PHENOMENON IS REAL ON OLD US_NEWS DEV, BUT RAW/PROXY SIGN INVERSION IS NOT ESTABLISHED. "
                "V69 supports agreement-aware US_SEC economic ranking, not a general polarity flip. All block signs remain "
                "diagnostic and require locked NEW DATA transfer plus the requested nested contribution-level counterfactual."
            ),
        },
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_projection(reconstruction: dict[str, Any]) -> dict[str, Any]:
    v64_summary = {}
    for source in ("US_NEWS", "US_SEC"):
        rows = [row for row in reconstruction["v64_direction_diagnostic_rows"] if row["source_family"] == source]
        v64_summary[source] = {
            "stable_positive": {
                row["feature_block"]: {
                    "stability": row["stability"],
                    "median_combined_contribution": row["median_combined_contribution"],
                }
                for row in rows if row["polarity"] == "+"
            },
            "stable_negative": {
                row["feature_block"]: {
                    "stability": row["stability"],
                    "median_combined_contribution": row["median_combined_contribution"],
                }
                for row in rows if row["polarity"] == "-"
            },
            "conservative_zero": [row["feature_block"] for row in rows if row["polarity"] == "0"],
        }
    confidence = reconstruction["confidence_reliability"]
    confidence_summary = {
        "authority": confidence["authority"],
        "implementation_caveat": confidence["implementation_caveat"],
        "us_news_high_confidence_reversal": confidence["us_news_high_confidence_reversal"],
        "sources": {},
    }
    for source in SOURCES:
        item = confidence["scopes"][source]
        frozen = item["frozen_v58_confidence"]
        margin = item["model_probability_margin"]
        confidence_summary["sources"][source] = {
            "assessment": item["assessment"],
            "confidence_block_polarity": item["confidence_role_block_polarity"],
            "n": frozen["n"],
            "frozen_v58_confidence": {
                key: frozen[key] for key in (
                    "confidence_correctness_auc", "top20_accuracy", "bottom20_accuracy",
                    "top20_minus_bottom20_accuracy", "accuracy_spearman_by_decile",
                    "frozen_highconf_accuracy", "frozen_non_highconf_accuracy",
                )
            },
            "model_probability_margin": {
                key: margin[key] for key in (
                    "confidence_correctness_auc", "top20_accuracy", "bottom20_accuracy",
                    "top20_minus_bottom20_accuracy", "accuracy_spearman_by_decile",
                )
            },
        }
    roles = {}
    for source, item in reconstruction["source_role_summary"].items():
        roles[source] = {
            "v67_source_n": item["v67_source_n"],
            "v67_source_supported": item["v67_source_supported"],
            "direction_nonzero_blocks": item["direction_nonzero_blocks"],
            "direction_zero_blocks": item["direction_zero_blocks"],
            "confidence_source_level_assessment": item["confidence_source_level_assessment"],
            "confidence_block_polarity": item["confidence_block_polarity"],
            "opportunity_nonzero_blocks": item["opportunity_nonzero_blocks"],
            "opportunity_zero_blocks": item["opportunity_zero_blocks"],
        }
    interventions = reconstruction["intervention_results"]
    v68 = interventions["V68"]
    v69 = interventions["V69"]
    intervention_summary = {
        "V65": interventions["V65"],
        "V66": interventions["V66"],
        "V68": {
            "scope": v68["scope"],
            "status": v68["status"],
            "v67_strict_opposite_blocks_tested": v68["v67_strict_opposite_blocks_tested"],
            "eligible_outer_oof_n": v68["eligible_outer_oof_n"],
            "baseline_direction": v68["baseline_direction"],
            "candidate_direction": v68["candidate_direction"],
            "baseline_opportunity": v68["baseline_opportunity"],
            "candidate_opportunity": v68["candidate_opportunity"],
            "direction_positive_fold_fraction": v68["direction_positive_fold_fraction"],
            "opportunity_positive_fold_fraction": v68["opportunity_positive_fold_fraction"],
            "fallback": v68["fallback"],
            "authoritative_result": v68["authoritative_result"],
        },
        "V69": {
            "scope": v69["scope"],
            "status": v69["status"],
            "candidate_contract": v69["candidate_contract"],
            "outer_evidence_rows": v69["outer_evidence_rows"],
            "outer_evidence_fraction_of_us_sec": v69["outer_evidence_fraction_of_us_sec"],
            "outer_baseline": {
                key: v69["outer_baseline"][key] for key in (
                    "n", "highconf_n", "highconf_coverage", "highconf_accuracy",
                    "highconf_mean_signed_net", "confidence_correctness_auc",
                )
            },
            "outer_candidate": {
                key: v69["outer_candidate"][key] for key in (
                    "n", "highconf_n", "highconf_coverage", "highconf_accuracy",
                    "highconf_mean_signed_net", "confidence_correctness_auc",
                )
            },
            "outer_delta": v69["outer_delta"],
            "outer_fold_net_deltas": v69["outer_fold_net_deltas"],
            "positive_fold_fraction": v69["positive_fold_fraction"],
            "nonnegative_fold_fraction": v69["nonnegative_fold_fraction"],
            "full_us_sec_highconf_net": v69["full_us_sec_highconf_net"],
            "bootstrap": {
                "draws": v69["bootstrap"]["draws"],
                "method": v69["bootstrap"]["method"],
                "net_delta_mean": v69["bootstrap"]["net_delta_mean"],
                "net_delta_probability_gt_zero": v69["bootstrap"]["net_delta_probability_gt_zero"],
                "net_delta_median": v69["bootstrap"]["net_delta_quantiles"]["0.5"],
                "net_delta_lower95": v69["bootstrap"]["net_delta_quantiles"]["0.025"],
            },
            "agreement_cohorts": v69["agreement_cohorts"],
            "direction_immutability": {
                key: v69["direction_immutability"][key] for key in (
                    "baseline_direction_sha256", "candidate_direction_sha256",
                    "baseline_probability_sha256", "candidate_probability_sha256",
                    "direction_mismatch_n", "max_abs_probability_delta", "exact_unchanged",
                )
            },
            "canonical_gate_qualification": v69["canonical_gate_qualification"],
            "authoritative_effect": v69["authoritative_effect"],
        },
    }
    return {
        "scope": reconstruction["scope"],
        "integrity": reconstruction["integrity"],
        "method_authority": reconstruction["method_authority"],
        "confidence_reliability": confidence_summary,
        "v64_direction_diagnostic_summary": v64_summary,
        "source_role_summary": roles,
        "role_divergence": reconstruction["role_divergence"],
        "v64_v67_contradictions": reconstruction["v64_v67_contradictions"],
        "intervention_results": intervention_summary,
        "evidence_boundary": reconstruction["evidence_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--no-expected", action="store_true")
    parser.add_argument("--print-report", action="store_true")
    parser.add_argument("--print-artifact", action="store_true")
    args = parser.parse_args()
    runtime = bounded_runtime()
    reconstruction = build_reconstruction()
    digest = canonical_sha256(reconstruction)
    audit_script_sha256 = sha256(Path(__file__).resolve())
    expected_match: bool | None = None
    expected_path: str | None = None
    if not args.no_expected:
        expected_path = str(args.expected.resolve())
        require(args.expected.is_file(), f"expected reconstruction missing: {args.expected}")
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        require("reconstruction" in expected, "expected artifact has no reconstruction field")
        expected_match = expected["reconstruction"] == artifact_projection(reconstruction)
        require(expected_match, "expected OLD polarity reconstruction does not match committed evidence")
        require(expected.get("reconstruction_sha256") == digest, "expected reconstruction digest changed")
        require(expected.get("audit_script_sha256") == audit_script_sha256, "expected audit script digest changed")
    output = {
        "status": "PASS",
        "audit_mode": "READ_ONLY",
        "output_written": False,
        "runtime": runtime,
        "audit_script_sha256": audit_script_sha256,
        "reconstruction_sha256": digest,
        "expected_path": expected_path,
        "expected_match": expected_match,
    }
    if args.print_report:
        output["reconstruction"] = reconstruction
    if args.print_artifact:
        output = {
            "schema_version": 1,
            "artifact_type": "V221_OLD_POLARITY_RECONSTRUCTION",
            "generated_by": Path(__file__).name,
            "audit_script_sha256": audit_script_sha256,
            "reconstruction_sha256": digest,
            "reconstruction": artifact_projection(reconstruction),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
