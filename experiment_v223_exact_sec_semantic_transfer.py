"""V223: exact-accession SEC semantic transfer on the frozen V221 epoch.

The candidate family and weights are fixed from pre-epoch V59 artifacts.  New
labels are used only after each block's predictions exist (and as causal past
labels for later blocks); they never select a candidate, C, blend or cutoff.
This is a transfer diagnostic and cannot open Seal/Final or replace V69.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import experiment_v37_text as v37
import experiment_v38_opportunity as v38
import experiment_v59_sec_structured_v1 as v59
import runtime_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESEARCH = ROOT / "research"
OUT = ROOT / "output_V223"
LEGACY = DATA / "dev_contract_v36_labeled.csv.gz"
EXTENSION = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"
EPOCH = RESEARCH / "NEW_EXACT_DEV_EXTENSION_DATA_EPOCH_CONTRACT.json"
TEXT_MANIFEST = RESEARCH / "V221_US_SEC_EXACT_ACCESSION_TEXT_MANIFEST.csv"
V59_COMPARISON = ROOT / "output_V59" / "MODEL_COMPARISON.json"
EXPECTED_EXTENSION_SHA256 = "e11c60ef38c5b65e676bab79c96cd21e44fdcb6d604e02a550ffc49003ed311b"
EXPECTED_TEXT_MANIFEST_SHA256 = "96b2e108dcf673071b6520e261a96b832e4f5314c2397974a542bb108c939d89"
COST = 0.002

# Frozen before current-label evaluation. N_S was the strongest V59 component
# on pre-epoch DEV (BA/AUC/HC metrics), and C=.25 is the latest V59 setting.
PRIMARY_NAME = "V223_FROZEN_V59_N_S"
FROZEN_C = 0.25
FROZEN_CANDIDATES = {
    "V223_FROZEN_V59_N_S": {"N": 0.5, "S": 0.5},
    "V223_NUMERIC_CONTROL": {"N": 1.0},
    "V223_STRUCTURE_ONLY_DIAGNOSTIC": {"S": 1.0},
    "V223_FROZEN_V59_N_T_S_DIAGNOSTIC": {"N": 0.5, "T": 0.25, "S": 0.25},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    atomic_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", path)


def atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            compressed.write(raw)
    os.replace(temporary, path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if sha256(EXTENSION) != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError("V223_EXTENSION_HASH_MISMATCH")
    if sha256(TEXT_MANIFEST) != EXPECTED_TEXT_MANIFEST_SHA256:
        raise RuntimeError("V223_EXACT_TEXT_MANIFEST_HASH_MISMATCH")
    epoch = json.loads(EPOCH.read_text(encoding="utf-8"))
    if epoch.get("status") != "FROZEN" or epoch.get("reserved_labels_opened") is not False:
        raise RuntimeError("V223_FROZEN_EPOCH_GUARD_FAIL")
    historical = json.loads(V59_COMPARISON.read_text(encoding="utf-8"))
    legacy = pd.read_csv(LEGACY, compression="gzip", low_memory=False, dtype={"ticker": str},
                         parse_dates=["event_time_utc"])
    extension = pd.read_csv(EXTENSION, compression="gzip", low_memory=False, dtype={"ticker": str},
                            parse_dates=["event_time_utc"])
    extension = extension[extension.source_family.eq("US_SEC")].copy()
    manifest = pd.read_csv(TEXT_MANIFEST, dtype=str).set_index("event_id")
    if set(extension.event_id) != set(manifest.index) or not manifest.status.eq("PASS").all():
        raise RuntimeError("V223_EXACT_TEXT_IDENTITY_COVERAGE_FAIL")
    extension["body"] = extension.event_id.map(
        lambda event_id: (ROOT / manifest.at[event_id, "text_path"]).read_text(encoding="utf-8", errors="replace")
    )
    legacy["headline"] = legacy.headline.fillna("").astype(str)
    legacy["body"] = legacy.body.fillna("").astype(str)
    legacy, _ = v59.recover_bodies(legacy)
    extension["headline"] = extension.headline.fillna("").astype(str)
    extension["body"] = extension.body.fillna("").astype(str)
    for frame, origin in ((legacy, "V36_DEV"), (extension, "V221_DEV_EXTENSION")):
        frame["origin"] = origin
        frame["text_cluster_id"] = frame.apply(v37.normalized_cluster, axis=1)
    combined = pd.concat([legacy, extension], ignore_index=True, sort=False)
    sizes = combined.groupby("text_cluster_id").event_id.transform("size")
    combined["cluster_weight"] = 1.0 / sizes
    legacy = combined[combined.origin.eq("V36_DEV")].copy()
    extension = combined[combined.origin.eq("V221_DEV_EXTENSION")].copy()
    for frame in (legacy, extension):
        frame["sec_segment_text"] = frame.apply(lambda row: v59.segments(row)["combined"], axis=1)
        frame["sec_structure"] = frame.apply(v59.structured_features, axis=1)
        frame["body_present"] = frame.body.str.strip().ne("")
    return legacy, extension, epoch, historical


def extension_blocks(extension: pd.DataFrame, epoch: dict[str, Any]) -> list[pd.DataFrame]:
    definitions = epoch["counts"]["by_source_family"]["US_SEC"]["chronological_blocks"]
    output = []
    assigned: set[str] = set()
    for definition in definitions:
        start = pd.Timestamp(definition["start_utc"])
        end = pd.Timestamp(definition["end_utc"])
        block = extension[extension.event_time_utc.between(start, end, inclusive="both")].copy()
        block = block.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
        if len(block) != int(definition["n"]) or block.y.nunique() != 2:
            raise RuntimeError(f"V223_FROZEN_BLOCK_MISMATCH:{definition['block']}")
        assigned.update(block.event_id.astype(str))
        output.append(block)
    if assigned != set(extension.event_id.astype(str)):
        raise RuntimeError("V223_FROZEN_BLOCK_COVERAGE_FAIL")
    return output


def prediction_frame(valid: pd.DataFrame, probability: np.ndarray, fold: int, family: str,
                     seen: set[str]) -> pd.DataFrame:
    output = valid[["event_id", "event_group_id", "event_time_utc", "market", "ticker",
                    "source_family", "form", "y", "fwd_ret_30m", "text_cluster_id",
                    "cluster_weight", "body_present"]].copy()
    probability = np.asarray(probability, dtype=float)
    margin = np.abs(probability - 0.5)
    # Historical V36/V37/V59 convention: fixed top 20% confidence coverage.
    threshold = float(np.quantile(margin, 0.80))
    output["prob"] = probability
    output["high_conf"] = margin >= threshold
    output["confidence_signal"] = margin
    output["fold"] = fold
    output["family"] = family
    output["new_issuer"] = ~output.ticker.astype(str).isin(seen)
    return output


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame.y.to_numpy(int)
    probability = frame.prob.to_numpy(float)
    prediction = probability >= 0.5
    high = frame.high_conf.to_numpy(bool)
    signed = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    prevalence = float(y.mean())
    return {
        "n": int(len(frame)),
        "y_0": int((y == 0).sum()),
        "y_1": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auc": float(roc_auc_score(y, probability)),
        "edge_vs_naive": float(accuracy_score(y, prediction) - max(prevalence, 1.0 - prevalence)),
        "pred_up": float(prediction.mean()),
        "highconf_n": int(high.sum()),
        "highconf_coverage": float(high.mean()),
        "highconf_accuracy": float((prediction[high] == y[high]).mean()),
        "highconf_net": float(signed[high].mean()),
        "all_trade_net": float(signed.mean()),
        "body_coverage": float(frame.body_present.mean()),
    }


def main() -> int:
    runtime_limits.configure()
    legacy, extension, epoch, historical = load_data()
    blocks = extension_blocks(extension, epoch)
    pieces = {name: [] for name in FROZEN_CANDIDATES}
    audits = []
    for fold, valid in enumerate(blocks, start=1):
        boundary = valid.event_time_utc.min()
        causal_cutoff = boundary - pd.Timedelta(minutes=35)
        train_legacy = legacy[(legacy.market.eq("US")) & (legacy.event_time_utc < causal_cutoff)].copy()
        train_extension = extension[extension.event_time_utc < causal_cutoff].copy()
        train = pd.concat([train_legacy, train_extension], ignore_index=True, sort=False)
        overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
        if overlap:
            raise RuntimeError(f"V223_EVENT_GROUP_LEAKAGE:{fold}:{len(overlap)}")
        train_sec = train[train.source_family.eq("US_SEC")].copy()
        if len(train) < 200 or len(train_sec) < 200 or train.y.nunique() != 2:
            raise RuntimeError(f"V223_CAUSAL_TRAIN_INSUFFICIENT:{fold}")
        numeric = v38.target_fit(train, valid, "y")
        text, structure = v59.fit_predict(train_sec, valid, FROZEN_C)
        parts = {"N": numeric, "T": text, "S": structure}
        seen = set(train.ticker.astype(str))
        for name, weights in FROZEN_CANDIDATES.items():
            probability = v59.blend(parts, weights)
            pieces[name].append(prediction_frame(valid, probability, fold, name, seen))
        audits.append({
            "fold": fold,
            "boundary_utc": boundary,
            "causal_cutoff_utc": causal_cutoff,
            "valid_n": len(valid),
            "valid_y_0": int((valid.y == 0).sum()),
            "valid_y_1": int((valid.y == 1).sum()),
            "train_n": len(train),
            "train_legacy_n": len(train_legacy),
            "train_prior_extension_n": len(train_extension),
            "train_sec_n": len(train_sec),
            "train_end_utc": train.event_time_utc.max(),
            "event_group_overlap": 0,
            "valid_body_coverage": float(valid.body_present.mean()),
        })
        print(f"[V223] fold={fold}/5 train={len(train)} sec={len(train_sec)} valid={len(valid)}", flush=True)
    predictions = {name: pd.concat(value, ignore_index=True).sort_values("event_time_utc")
                   for name, value in pieces.items()}
    reports = {}
    for name, frame in predictions.items():
        reports[name] = {
            "overall": metrics(frame),
            "by_block": {str(int(fold)): metrics(group) for fold, group in frame.groupby("fold")},
            "by_form": {str(form): metrics(group) for form, group in frame.groupby("form")
                        if len(group) >= 8 and group.y.nunique() == 2},
        }
    primary = reports[PRIMARY_NAME]["overall"]
    control = reports["V223_NUMERIC_CONTROL"]["overall"]
    fixed_transfer_checks = {
        "balanced_accuracy_ge_0_540": primary["balanced_accuracy"] >= 0.540,
        "auc_ge_0_560": primary["auc"] >= 0.560,
        "edge_ge_0_020": primary["edge_vs_naive"] >= 0.020,
        "highconf_coverage_ge_0_15": primary["highconf_coverage"] >= 0.15,
        "highconf_accuracy_ge_0_600": primary["highconf_accuracy"] >= 0.600,
        "highconf_net_gt_0": primary["highconf_net"] > 0,
        "all_five_blocks_ba_ge_0_490": min(
            value["balanced_accuracy"] for value in reports[PRIMARY_NAME]["by_block"].values()
        ) >= 0.490,
        "beats_numeric_control_ba": primary["balanced_accuracy"] > control["balanced_accuracy"],
        "beats_numeric_control_auc": primary["auc"] > control["auc"],
    }
    passed = sum(fixed_transfer_checks.values())
    status = "TRANSFER_SURVIVOR" if all(fixed_transfer_checks.values()) else "RESEARCH_FAIL"
    report = {
        "version": 223,
        "hypothesis": "EXACT_ACCESSION_SEC_STRUCTURE_ADDS_CAUSAL_DIRECTION_INFORMATION",
        "status": status,
        "evaluation_type": "FROZEN_PRE_EPOCH_POLICY_PREQUENTIAL_TRANSFER",
        "primary_candidate": PRIMARY_NAME,
        "primary_selection_basis": {
            "source": "pre-epoch output_V59/MODEL_COMPARISON.json only",
            "reason": "N_S had the strongest frozen V59 component BA/AUC with positive HC net",
            "new_epoch_labels_used_for_selection": False,
            "C": FROZEN_C,
            "weights": FROZEN_CANDIDATES[PRIMARY_NAME],
            "direction_cutoff": 0.5,
            "confidence_rule": "label-blind top 20% absolute margin within each frozen block",
        },
        "results": reports,
        "primary_delta_vs_numeric": {
            "balanced_accuracy": primary["balanced_accuracy"] - control["balanced_accuracy"],
            "auc": primary["auc"] - control["auc"],
            "highconf_accuracy": primary["highconf_accuracy"] - control["highconf_accuracy"],
            "highconf_net": primary["highconf_net"] - control["highconf_net"],
        },
        "transfer_gate": {"passed": passed, "total": len(fixed_transfer_checks),
                          "checks": fixed_transfer_checks, "survivor": all(fixed_transfer_checks.values())},
        "canonical_14_gate_evaluated": False,
        "canonical_14_gate_reason": "single-source new-epoch transfer diagnostic; no KR or legacy OOF replacement",
        "fold_audits": audits,
        "data_audit": {
            "extension_rows": len(extension),
            "exact_body_rows": int(extension.body_present.sum()),
            "exact_body_coverage": float(extension.body_present.mean()),
            "text_manifest_sha256": sha256(TEXT_MANIFEST),
            "extension_sha256": sha256(EXTENSION),
            "epoch_sha256": epoch.get("data_epoch_sha256"),
            "legacy_sha256": sha256(LEGACY),
            "v59_comparison_sha256": sha256(V59_COMPARISON),
        },
        "guards": {
            "selection_uses_current_labels": False,
            "fold_training_uses_only_rows_35m_before_boundary": True,
            "reserved_research_seal_or_final_rows_read": False,
            "v69_champion_changed": False,
            "seal_authorized": False,
            "final_meta_authorized": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(report, OUT / "V223_EXACT_SEC_SEMANTIC_TRANSFER_REPORT.json")
    for name, frame in predictions.items():
        atomic_csv_gz(frame, OUT / f"{name}_PREDICTIONS.csv.gz")
    run_status = {
        "version": 223,
        "status": status,
        "phase": "NEW_DATA_RESEARCH",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seal_state": "UNOPENED",
        "final_status": "CONTINUE",
    }
    atomic_json(run_status, OUT / "RUN_STATUS.json")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(OUT.iterdir()) if path.is_file()
             and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}}
    atomic_json({"version": 223, "hypothesis": report["hypothesis"], "files": files,
                 "research_seal_or_final_rows_read": False}, OUT / "ARTIFACT_MANIFEST.json")
    atomic_json({
        "version": 223,
        "hypothesis": report["hypothesis"],
        "committed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
        "data_epoch_sha256": epoch.get("data_epoch_sha256"),
        "runner_sha256": sha256(Path(__file__)),
    }, OUT / "COMMIT.json")
    print(json.dumps({
        "version": 223,
        "status": status,
        "transfer_gate": report["transfer_gate"],
        "primary": primary,
        "numeric_control": control,
        "delta": report["primary_delta_vs_numeric"],
        "commit_sha256": sha256(OUT / "COMMIT.json"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
