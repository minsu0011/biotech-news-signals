"""V226 material follow-up: US_NEWS P(correct) inference/opportunity gates.

Uses the V225 FULL_STRUCTURED correctness model with an inner-past 75th
percentile threshold, then combines it with the independently cross-fitted
V72 P(material move) and its already-frozen per-fold inner threshold.
Direction probabilities are never changed.  This is current-DEV research and
cannot be independent confirmation or authorize Seal/Final.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import experiment_v225_high_confidence_news_discriminator as v225


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research" / "v224_news_confidence" / "V226_US_DUAL_GATE"
V72 = ROOT / "research" / "V72_US_NEWS_DUAL_GATE_STANDALONE" / "V72_DUAL_GATE_OUTER_OOF.csv.gz"
EXPECTED_V72_SHA = "a239d6cf3bf413a47dcf01da0d32666f420403693b11c4c8e518b78b86387bf2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gate_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    selected = frame[column].astype(bool).to_numpy()
    net = frame.loc[selected, "signed_net"].to_numpy(float)
    ordered = np.sort(net)
    work = frame.copy().reset_index(drop=True)
    work["news_high_conf"] = selected
    bootstrap = v225.bootstrap_net(work, draws=2000)
    return {
        "n": len(frame), "selected_n": int(selected.sum()), "coverage": float(selected.mean()),
        "accuracy": float(frame.loc[selected, "correct_target"].mean()),
        "mean_net": float(net.mean()),
        "winsorized_net": float(np.clip(net, *np.quantile(net, [0.01, 0.99])).mean()),
        "top1_removed_net": float(ordered[:-1].mean()),
        "top5_removed_net": float(ordered[:-5].mean()),
        "bootstrap_net": bootstrap,
    }


def main() -> int:
    if sha256(V72) != EXPECTED_V72_SHA:
        raise RuntimeError("V226_V72_HASH_MISMATCH")
    authority = v225.verify_authority()
    news = v225.load_frame()
    us = news[news.source_family.eq("US_NEWS")].copy()
    # Material dosage experiment: increase P(correct) coverage from the V225
    # 80th-percentile baseline to a fixed inner-past 75th percentile.
    original_quantile = v225.HC_QUANTILE
    v225.HC_QUANTILE = 0.75
    try:
        predictions, audits = v225.run_family(us, "V226_FULL_STRUCTURED_Q75", "full", False)
    finally:
        v225.HC_QUANTILE = original_quantile
    material = pd.read_csv(V72, compression="gzip", low_memory=False,
                           usecols=["event_id", "p_material_move", "m_cutoff"])
    evidence = predictions.merge(material, on="event_id", how="left", validate="one_to_one")
    if evidence[["p_material_move", "m_cutoff"]].isna().any().any():
        raise RuntimeError("V226_V72_EVENT_COVERAGE_FAIL")
    evidence["correctness_gate"] = evidence.news_high_conf.astype(bool)
    evidence["dual_gate"] = evidence.correctness_gate & (evidence.p_material_move >= evidence.m_cutoff)
    inference = gate_metrics(evidence, "correctness_gate")
    dual = gate_metrics(evidence, "dual_gate")
    checks = {
        "direction_probability_unchanged": True,
        "coverage_ge_0_15": dual["coverage"] >= 0.15,
        "hc_accuracy_ge_0_60": dual["accuracy"] >= 0.60,
        "hc_net_gt_0": dual["mean_net"] > 0,
        "bootstrap_hc_net_lower_gt_0": dual["bootstrap_net"]["lower95"] > 0,
        "winsorized_net_gt_0": dual["winsorized_net"] > 0,
        "top5_removed_net_gt_0": dual["top5_removed_net"] > 0,
        "strict_nested_fold_audits": all(row["text_cluster_overlap"] == 0 and
                                           row["outer_labels_used_for_fit_or_threshold"] is False
                                           for row in audits),
    }
    report = {
        "version": 226,
        "hypothesis": "US_NEWS_CORRECTNESS_PLUS_OPPORTUNITY_DUAL_GATE_Q75",
        "status": "RESEARCH_SURVIVOR" if all(checks.values()) else "RESEARCH_FAIL",
        "authority": authority,
        "contract": {
            "direction": "immutable V69 OOF",
            "p_correct": "V225 FULL_STRUCTURED, source-specific strict OOF",
            "correctness_threshold": "fixed inner-past 75th percentile",
            "p_opportunity": "pinned V72 strict-OOF P(abs(return)>=0.005)",
            "opportunity_threshold": "pinned V72 inner-selected m_cutoff",
            "outer_labels_used_for_threshold": False,
            "selection_bias": "Q75 material dosage chosen after V225 DEV analysis",
        },
        "inference_gate": inference,
        "dual_gate": dual,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "fold_audits": audits,
        "research_seal_or_final_rows_read": False,
        "seal_authorized": False,
        "final_meta_authorized": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(report, OUT / "V226_US_NEWS_DUAL_GATE_REPORT.json")
    v225.atomic_csv_gz(evidence, OUT / "V226_US_NEWS_DUAL_GATE_OOF.csv.gz")
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "MANIFEST.json"}
    atomic_json({"version": 226, "files": files, "runner_sha256": sha256(Path(__file__)),
                 "research_seal_or_final_rows_read": False}, OUT / "MANIFEST.json")
    print(json.dumps({"status": report["status"], "checks": checks,
                      "inference_gate": inference, "dual_gate": dual},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
