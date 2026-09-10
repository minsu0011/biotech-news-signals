"""V224: causal polarity/calibration/confidence guard over V223.

Every row-level decision for a block uses only predictions and outcomes from
strictly earlier frozen blocks.  The guard family was discovered during the
current DEV research loop, so a passing DEV transfer gate is explicitly not
an independent confirmation and cannot authorize Seal/Final.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import experiment_v223_exact_sec_semantic_transfer as v223
import experiment_v36_robust as v36


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_V224"
INPUTS = {
    "V224_EXACT_SEC_N_T_S_GUARDED": (
        ROOT / "output_V223" / "V223_FROZEN_V59_N_T_S_DIAGNOSTIC_PREDICTIONS.csv.gz",
        "85d3382bce4e2469d506e62e7e59feca011ba9d84a025e3a982dedc18d57e916",
    ),
    "V224_NUMERIC_GUARDED_CONTROL": (
        ROOT / "output_V223" / "V223_NUMERIC_CONTROL_PREDICTIONS.csv.gz",
        "45e202f06099a813b97d70f803f5d4e27297ed59428a36921e55b6709381d0c0",
    ),
}
V223_COMMIT = ROOT / "output_V223" / "COMMIT.json"
EXPECTED_V223_COMMIT_SHA256 = "3cd1128fb1bc0893f4f8404b935f730f3a9836c3cdae817a387e41de26cda7bc"
POLARITY_AUC_FLOOR = 0.48
CUTOFF_GRID = np.arange(0.30, 0.701, 0.01)
CONFIDENCE_QUANTILE = 0.80
COST = 0.002


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_cutoff(previous: pd.DataFrame) -> float:
    trials = []
    for cutoff in CUTOFF_GRID:
        accuracy = float(((previous.oriented_prob >= cutoff) == previous.y).mean())
        trials.append((accuracy, -abs(float(cutoff) - 0.5), float(cutoff)))
    return max(trials)[2]


def confidence_score(previous: pd.DataFrame, low: bool) -> tuple[int, float]:
    margin = previous.margin.to_numpy(float)
    threshold = float(np.quantile(margin, 1.0 - CONFIDENCE_QUANTILE if low else CONFIDENCE_QUANTILE))
    selected = margin <= threshold if low else margin >= threshold
    prediction = previous.prediction.to_numpy(bool)[selected]
    y = previous.y.to_numpy(int)[selected]
    returns = previous.fwd_ret_30m.to_numpy(float)[selected]
    accuracy = float((prediction == y).mean())
    net = float((np.where(prediction, 1.0, -1.0) * returns - COST).mean())
    checks = int(accuracy >= 0.60) + int(net > 0)
    return checks, accuracy + 5.0 * float(np.clip(net, -0.01, 0.01))


def apply_guard(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    raw = frame.sort_values(["fold", "event_time_utc", "event_id"]).reset_index(drop=True)
    completed: list[pd.DataFrame] = []
    audits = []
    for fold in sorted(raw.fold.unique()):
        current = raw[raw.fold.eq(fold)].copy()
        causal_raw = raw[raw.fold.lt(fold)].copy()
        prior_auc = None
        reverse = False
        if len(causal_raw) and causal_raw.y.nunique() == 2:
            prior_auc = float(roc_auc_score(causal_raw.y, causal_raw.prob))
            reverse = prior_auc < POLARITY_AUC_FLOOR
        current["oriented_prob"] = 1.0 - current.prob if reverse else current.prob
        cutoff = 0.5 if not completed else select_cutoff(completed[-1])
        current["prob"] = np.clip(0.5 + current.oriented_prob - cutoff, 0.0, 1.0)
        current["prediction"] = current.prob >= 0.5
        current["margin"] = np.abs(current.prob - 0.5)
        use_low_margin = False
        high_score = low_score = None
        if completed:
            high_score = confidence_score(completed[-1], low=False)
            low_score = confidence_score(completed[-1], low=True)
            use_low_margin = low_score > high_score
        threshold = float(np.quantile(
            current.margin,
            1.0 - CONFIDENCE_QUANTILE if use_low_margin else CONFIDENCE_QUANTILE,
        ))
        current["high_conf"] = current.margin <= threshold if use_low_margin else current.margin >= threshold
        current["confidence_signal"] = -current.margin if use_low_margin else current.margin
        current["polarity_reversed"] = reverse
        current["direction_cutoff"] = cutoff
        current["confidence_polarity"] = "LOW_MARGIN" if use_low_margin else "HIGH_MARGIN"
        audits.append({
            "fold": int(fold),
            "prior_raw_rows": int(len(causal_raw)),
            "prior_raw_auc": prior_auc,
            "polarity_auc_floor": POLARITY_AUC_FLOOR,
            "polarity_reversed": reverse,
            "calibration_source": "immediately_previous_completed_block" if completed else "pre_epoch_default",
            "direction_cutoff": cutoff,
            "previous_high_margin_score": high_score,
            "previous_low_margin_score": low_score,
            "confidence_polarity": "LOW_MARGIN" if use_low_margin else "HIGH_MARGIN",
            "confidence_threshold": threshold,
            "current_labels_used_for_any_decision": False,
        })
        completed.append(current)
    return pd.concat(completed, ignore_index=True), audits


def bootstrap_summary(frame: pd.DataFrame) -> dict[str, Any]:
    view = frame.copy()
    view["new_issuer"] = view.new_issuer.astype(bool)
    return v36.bootstrap(view, nboot=2000)


def main() -> int:
    if sha256(V223_COMMIT) != EXPECTED_V223_COMMIT_SHA256:
        raise RuntimeError("V224_V223_COMMIT_HASH_MISMATCH")
    outputs = {}
    audits = {}
    for name, (path, expected) in INPUTS.items():
        if sha256(path) != expected:
            raise RuntimeError(f"V224_INPUT_HASH_MISMATCH:{name}")
        frame = pd.read_csv(path, compression="gzip", low_memory=False,
                            parse_dates=["event_time_utc"])
        guarded, audit = apply_guard(frame)
        outputs[name] = guarded
        audits[name] = audit
    primary_name = "V224_EXACT_SEC_N_T_S_GUARDED"
    control_name = "V224_NUMERIC_GUARDED_CONTROL"
    results = {}
    for name, frame in outputs.items():
        results[name] = {
            "overall": v223.metrics(frame),
            "by_block": {str(int(fold)): v223.metrics(group) for fold, group in frame.groupby("fold")},
            "bootstrap": bootstrap_summary(frame),
        }
    primary = results[primary_name]["overall"]
    control = results[control_name]["overall"]
    checks = {
        "balanced_accuracy_ge_0_540": primary["balanced_accuracy"] >= 0.540,
        "auc_ge_0_560": primary["auc"] >= 0.560,
        "edge_ge_0_020": primary["edge_vs_naive"] >= 0.020,
        "highconf_coverage_ge_0_15": primary["highconf_coverage"] >= 0.15,
        "highconf_accuracy_ge_0_600": primary["highconf_accuracy"] >= 0.600,
        "highconf_net_gt_0": primary["highconf_net"] > 0,
        "all_five_blocks_ba_ge_0_490": min(
            value["balanced_accuracy"] for value in results[primary_name]["by_block"].values()
        ) >= 0.490,
        "semantic_ba_delta_vs_guarded_numeric_ge_0_020": (
            primary["balanced_accuracy"] - control["balanced_accuracy"] >= 0.020
        ),
        "semantic_edge_delta_vs_guarded_numeric_ge_0_020": (
            primary["edge_vs_naive"] - control["edge_vs_naive"] >= 0.020
        ),
    }
    bootstrap_checks = {
        "bootstrap_ba_lower_ge_0_500": (
            results[primary_name]["bootstrap"]["balanced_accuracy_lower95"] >= 0.500
        ),
        "bootstrap_highconf_net_lower_gt_0": (
            results[primary_name]["bootstrap"]["highconf_strategy_net_lower95"] > 0
        ),
    }
    gate_pass = all(checks.values())
    status = "DEV_RESEARCH_SURVIVOR_PENDING_INDEPENDENT_CONFIRMATION" if gate_pass else "RESEARCH_FAIL"
    report = {
        "version": 224,
        "hypothesis": "CAUSAL_DIRECTION_AND_CONFIDENCE_POLARITY_ADAPTATION",
        "status": status,
        "primary_candidate": primary_name,
        "results": results,
        "causal_policy": {
            "polarity": "reverse current block iff cumulative strictly-prior raw AUC < 0.48",
            "calibration": "cutoff maximizing accuracy on immediately prior completed block; fixed 0.30..0.70 grid",
            "confidence": "choose high- vs low-margin 20% using immediately prior block only",
            "current_block_labels_used": False,
            "current_block_returns_used": False,
        },
        "fold_audits": audits,
        "transfer_gate": {"passed": sum(checks.values()), "total": len(checks),
                          "checks": checks, "survivor": gate_pass},
        "bootstrap_confirmation_checks": bootstrap_checks,
        "primary_delta_vs_guarded_numeric": {
            "balanced_accuracy": primary["balanced_accuracy"] - control["balanced_accuracy"],
            "auc": primary["auc"] - control["auc"],
            "edge": primary["edge_vs_naive"] - control["edge_vs_naive"],
            "highconf_accuracy": primary["highconf_accuracy"] - control["highconf_accuracy"],
            "highconf_net": primary["highconf_net"] - control["highconf_net"],
        },
        "selection_bias_disclosure": {
            "guard_family_discovered_on_current_dev": True,
            "independent_confirmation_complete": False,
            "eligible_for_seal": False,
            "eligible_for_final": False,
            "reason": "V224 architecture-level selection followed inspection of V223 current-DEV diagnostics",
        },
        "guards": {
            "research_seal_or_final_rows_read": False,
            "v69_champion_changed": False,
            "seal_authorized": False,
            "final_meta_authorized": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(report, OUT / "V224_CAUSAL_POLARITY_CONFIDENCE_GUARD_REPORT.json")
    for name, frame in outputs.items():
        v223.atomic_csv_gz(frame, OUT / f"{name}_PREDICTIONS.csv.gz")
    atomic_json({
        "version": 224,
        "status": status,
        "phase": "NEW_DATA_RESEARCH",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seal_state": "UNOPENED",
        "final_status": "CONTINUE",
    }, OUT / "RUN_STATUS.json")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(OUT.iterdir()) if path.is_file()
             and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}}
    atomic_json({"version": 224, "hypothesis": report["hypothesis"], "files": files,
                 "research_seal_or_final_rows_read": False}, OUT / "ARTIFACT_MANIFEST.json")
    atomic_json({
        "version": 224,
        "hypothesis": report["hypothesis"],
        "committed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
        "v223_commit_sha256": sha256(V223_COMMIT),
        "runner_sha256": sha256(Path(__file__)),
    }, OUT / "COMMIT.json")
    print(json.dumps({
        "version": 224,
        "status": status,
        "transfer_gate": report["transfer_gate"],
        "bootstrap_confirmation_checks": bootstrap_checks,
        "primary": primary,
        "guarded_numeric": control,
        "delta": report["primary_delta_vs_guarded_numeric"],
        "commit_sha256": sha256(OUT / "COMMIT.json"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
