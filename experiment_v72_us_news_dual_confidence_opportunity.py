"""V72 standalone US_NEWS dual confidence/opportunity research.

This experiment is deliberately non-canonical.  It reads only the immutable
V36 DEV contract and the atomically committed V69 champion.  V69 direction
probabilities and predictions are frozen.  For US_NEWS only, two separate
chronological cross-fitted models estimate P(V69 direction is correct) and
P(abs(30m return) >= 50bp).  A trade is eligible only when both estimates
clear a coarse policy selected on an inner-past validation slice.

The script never opens DEV_EXTENSION, role assignments, research seal, or
final-reserve data.  A challenger is applied only if every predeclared
material check passes; otherwise the selected frame is an exact V69 no-op.
The canonical 14-check research gate is recomputed from raw selected-frame
metrics by experiment_v44_microstructure.summarize, not accepted from a
runner-supplied count.  Outputs under research/V72_* are DRAFT evidence and
are not a registry entry, controller commit, or seal authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import experiment_v44_microstructure as v44
import experiment_v37_text as v37
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
DEFAULT_OUT = ROOT / "research" / "V72_US_NEWS_DUAL_GATE_STANDALONE"
STANDALONE_CACHE = DEFAULT_OUT
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")

VERSION = 72
HYPOTHESIS = "US_NEWS_DUAL_CORRECTNESS_OPPORTUNITY_GATE_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V36_ROWS = 9462
EXPECTED_V69_ROWS = 7568
EXPECTED_US_NEWS_ROWS = 954
EMBARGO = pd.Timedelta(minutes=35)
OUTER_BLOCKS = 6
INNER_TRAIN_FRACTION = 0.70
MATERIAL_MOVE_ABS_RETURN = 0.005
COST = 0.002
SEED = 7201
BOOTSTRAP_DRAWS = 2000
EXPECTED_STANDALONE_MANIFEST_SHA256 = "71bdee26cbc478f31c871b396678481e189119fe19236c4116afc1572db14d13"
C_QUANTILES = (0.45, 0.55, 0.65, 0.75)
M_QUANTILES = (0.35, 0.50, 0.65)
MIN_POLICY_COVERAGE = 0.07
MAX_POLICY_COVERAGE = 0.30

EXPERT_PROBABILITY_COLUMNS = (
    "prob",
    "prob_opportunity_direction",
    "prob_event_form_direction",
    "prob_fixed_numeric_direction",
)
EXPERT_CONFIDENCE_COLUMNS = (
    "confidence_signal",
    "opportunity_confidence",
    "confidence_opportunity_direction",
    "confidence_event_form_direction",
    "confidence_fixed_numeric_direction",
)
STATE_COLUMNS = (
    "pre_ret_2", "pre_ret_5", "pre_ret_15", "pre_ret_30", "pre_ret_60",
    "pre_vol_5", "pre_vol_10", "pre_vol_30", "pre_vol_60",
    "volume_ratio_1_30", "volume_ratio_5_30", "volume_ratio_10_60",
    "range_5m", "range_10m", "range_30m", "range_60m",
    "close_position_10", "close_position_30", "entry_bar_ret",
    "entry_bar_range", "intraday_ret_open", "return_autocorr_30",
    "up_fraction_10", "up_fraction_30", "trend_slope_30",
    "trend_slope_60", "vwap_distance_30", "minutes_from_open",
    "minutes_to_close", "event_positive_kw", "event_negative_kw",
    "event_financing_kw", "event_trial_kw", "event_regulatory_kw",
    "event_ma_kw", "event_earnings_kw", "headline_len", "body_len",
    "benchmark_ret_5", "benchmark_ret_15", "benchmark_ret_30",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, int)
    return float(roc_auc_score(target, score)) if np.unique(target).size == 2 else None


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.map(lambda item: str(item).strip().lower() in {"1", "true", "yes"}).astype(bool)


def verify_v69_authority() -> dict[str, Any]:
    commit_path = V69_DIR / "COMMIT.json"
    manifest_path = V69_DIR / "ARTIFACT_MANIFEST.json"
    require(commit_path.is_file() and manifest_path.is_file(), "V69 atomic authority is incomplete")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment changed")
    require(commit.get("manifest_sha256") == sha256(manifest_path), "V69 manifest hash mismatch")
    require(manifest.get("version") == 69, "V69 artifact manifest version mismatch")
    require(manifest.get("experiment_id") == commit.get("experiment_id"), "V69 authority experiment mismatch")
    for name, item in manifest.get("files", {}).items():
        path = V69_DIR / name
        require(path.is_file(), f"V69 manifested artifact missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V69 artifact size changed: {name}")
        require(sha256(path) == item["sha256"], f"V69 artifact hash changed: {name}")
    require(V69_PATH.name in manifest.get("files", {}), "V69 selected OOF is not manifested")
    dev_sha = sha256(DEV_PATH)
    require(dev_sha == EXPECTED_V36_SHA256, "immutable V36 DEV hash changed")
    require(commit.get("data_sha") == dev_sha, "V69 COMMIT is not bound to immutable V36 DEV")
    return {
        "authority": "output_V69 atomic COMMIT + ARTIFACT_MANIFEST",
        "commit_sha256": sha256(commit_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_files_verified": len(manifest["files"]),
        "experiment_id": commit["experiment_id"],
        "v36_dev_sha256": dev_sha,
        "v69_selected_oof_sha256": sha256(V69_PATH),
        "authorized_inputs": [
            str(DEV_PATH.relative_to(ROOT)),
            str(commit_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)),
            str(V69_PATH.relative_to(ROOT)),
        ],
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "research_seal_loaded": False,
        "final_reserve_loaded": False,
    }


def verify_standalone_cache(authority: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and load the immutable standalone evaluation without refitting."""
    manifest_path = STANDALONE_CACHE / "V72_ARTIFACT_MANIFEST.json"
    require(manifest_path.is_file(), "V72 standalone manifest is missing")
    manifest_sha = sha256(manifest_path)
    require(
        manifest_sha == EXPECTED_STANDALONE_MANIFEST_SHA256,
        "V72 standalone manifest changed; deterministic retraining is forbidden",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("version") == VERSION, "V72 standalone manifest version mismatch")
    require(manifest.get("hypothesis") == HYPOTHESIS, "V72 standalone hypothesis mismatch")
    require(manifest.get("noncanonical_draft") is True, "V72 cache is not marked draft")
    require(
        manifest.get("authority_experiment_id") == authority["experiment_id"],
        "V72 standalone cache is bound to another champion",
    )
    verified_files: dict[str, Any] = {}
    for name, item in manifest.get("files", {}).items():
        path = STANDALONE_CACHE / name
        require(path.is_file(), f"V72 cached artifact missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V72 cached artifact size changed: {name}")
        actual = sha256(path)
        require(actual == item["sha256"], f"V72 cached artifact hash changed: {name}")
        verified_files[name] = {"bytes": path.stat().st_size, "sha256": actual}
    required = {
        "V72_DIRECTION_IMMUTABILITY_AUDIT.json",
        "V72_DUAL_GATE_OUTER_OOF.csv.gz",
        "V72_FAIL_CLOSED_SELECTED_OOF.csv.gz",
        "V72_RUN_STATUS.json",
        "V72_STANDALONE_REPORT.json",
    }
    require(set(verified_files) == required, "V72 standalone cache file contract changed")
    report = json.loads((STANDALONE_CACHE / "V72_STANDALONE_REPORT.json").read_text(encoding="utf-8"))
    run_status = json.loads((STANDALONE_CACHE / "V72_RUN_STATUS.json").read_text(encoding="utf-8"))
    direction = json.loads(
        (STANDALONE_CACHE / "V72_DIRECTION_IMMUTABILITY_AUDIT.json").read_text(encoding="utf-8")
    )
    require(report.get("version") == VERSION and report.get("hypothesis") == HYPOTHESIS,
            "V72 standalone report identity mismatch")
    require(report.get("status") == "MATERIAL_FAIL_EXACT_V69_FALLBACK",
            "V72 standalone result is not the pinned fail-closed result")
    require(report.get("evaluation", {}).get("material_pass") is False,
            "V72 standalone material result changed")
    require(report.get("evaluation", {}).get("passed") == 9, "V72 material passed-count changed")
    require(report.get("evaluation", {}).get("total") == 14, "V72 material total-count changed")
    require(run_status.get("canonical_commit") is False, "standalone cache claims a canonical commit")
    require(run_status.get("material_pass") is False, "standalone run status changed")
    require(direction.get("probability_mismatch_n") == 0, "cached direction probability mismatch")
    require(direction.get("direction_mismatch_n") == 0, "cached direction mismatch")
    require(direction.get("non_us_news_confidence_mismatch_n") == 0,
            "cached non-US_NEWS confidence mismatch")
    require(direction.get("non_us_news_high_conf_mismatch_n") == 0,
            "cached non-US_NEWS high-confidence mismatch")
    champion = pd.read_csv(V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    cached_fallback = pd.read_csv(
        STANDALONE_CACHE / "V72_FAIL_CLOSED_SELECTED_OOF.csv.gz",
        compression="gzip", low_memory=False, dtype={"ticker": str},
    )
    require(champion.equals(cached_fallback), "cached fail-closed frame is not exact committed V69")
    return report, {
        "mode": "IMMUTABLE_STANDALONE_CACHE_ADOPTION",
        "retraining_performed": False,
        "standalone_cache": str(STANDALONE_CACHE.relative_to(ROOT)),
        "standalone_manifest_sha256": manifest_sha,
        "standalone_files_verified": verified_files,
        "v69_authority_reverified": True,
        "v36_sha256_reverified": authority["v36_dev_sha256"],
        "fail_closed_frame_exact_v69": True,
    }


def adopt_standalone_cache(output: Path, authority: dict[str, Any]) -> dict[str, Any]:
    """Emit controller-consumable runner reports from pinned standalone evidence."""
    report, cache_audit = verify_standalone_cache(authority)
    v69_robustness_path = V69_DIR / "DEV_ROBUSTNESS_REPORT.json"
    require(v69_robustness_path.is_file(), "committed V69 robustness report is missing")
    v69_robustness = json.loads(v69_robustness_path.read_text(encoding="utf-8"))
    require(v69_robustness.get("version") == "V69", "committed V69 report version mismatch")
    require(v69_robustness.get("status") == "MATERIAL_PASS", "unexpected V69 champion status")
    require(v69_robustness.get("selected", {}).get("research_gate", {}).get("passed") == 9,
            "V69 canonical gate changed")
    selected = json.loads(json.dumps(v69_robustness["selected"]))
    evaluation = report["evaluation"]
    selected["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": evaluation["passed"],
        "total": evaluation["total"],
        "material_pass": False,
        "evidence": str((STANDALONE_CACHE / "V72_STANDALONE_REPORT.json").relative_to(ROOT)),
        "standalone_manifest_sha256": cache_audit["standalone_manifest_sha256"],
    }
    selected["selection"] = {
        "policy": "EXACT_COMMITTED_V69_FALLBACK",
        "champion_experiment_id": authority["experiment_id"],
        "retraining_performed": False,
    }
    validation = (
        "Adopt immutable standalone V72 outer-OOF evidence after full manifest verification; "
        "no refit; committed V69 selected summary is the exact canonical fallback; "
        "35-minute chronological embargo and inner-only coarse c/m policy are documented in cache."
    )
    robustness = {
        "version": "V72",
        "hypothesis": HYPOTHESIS,
        "status": "RESEARCH_FAIL",
        "opened_dev_only": True,
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "seal_outcomes_loaded": False,
        "final_reserve_loaded": False,
        "selected": selected,
        "candidate": evaluation["candidate"],
        "v69_same_window_baseline": evaluation["v69_baseline"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_committed_v69": True,
        "cache_adoption": cache_audit,
        "seal_authorized": False,
    }
    comparison = {
        "version": "V72",
        "hypothesis": HYPOTHESIS,
        "status": "RESEARCH_FAIL",
        "selected_policy": "EXACT_COMMITTED_V69_FALLBACK",
        "models": {
            "V69_COMMITTED_CHAMPION": v69_robustness["selected"],
            "V72_DUAL_GATE_STANDALONE_DIAGNOSTIC": {
                "candidate": evaluation["candidate"],
                "v69_same_window_baseline": evaluation["v69_baseline"],
                "p_correct_correctness_auc": evaluation["p_correct_correctness_auc"],
                "p_material_move_auc": evaluation["p_material_move_auc"],
                "bootstrap": evaluation["bootstrap"],
                "material_gate": selected["material_gate"],
            },
            "V72_FAIL_CLOSED_SELECTED": selected,
        },
        "input_audit": authority,
        "cache_adoption": cache_audit,
        "validation": validation,
        "seal_authorized": False,
    }
    metrics = selected["metrics"]
    transfer = {
        "version": "V72",
        "hypothesis": HYPOTHESIS,
        "status": "RESEARCH_FAIL",
        "selected_oof_by_source": metrics["by_source_family"],
        "selected_oof_by_market": metrics["by_market"],
        "selected_oof_new_issuer": metrics["new_issuer"],
        "scope": "US_NEWS confidence/opportunity diagnostic; exact V69 selected fallback",
        "direction_exact_unchanged": True,
        "all_non_us_news_rows_unchanged": True,
        "retraining_performed": False,
        "method": validation,
        "seal_outcomes_loaded": False,
    }
    direction = json.loads(
        (STANDALONE_CACHE / "V72_DIRECTION_IMMUTABILITY_AUDIT.json").read_text(encoding="utf-8")
    )
    diagnostic = {
        "version": "V72",
        "hypothesis": HYPOTHESIS,
        "status": "CACHE_ADOPTED_RESEARCH_FAIL",
        "cache_adoption": cache_audit,
        "authority_audit": authority,
        "material_evaluation": evaluation,
        "direction_immutability": direction,
        "selected_policy": "EXACT_COMMITTED_V69_FALLBACK",
        "canonical_commit_created_by_runner": False,
        "controller_owns_plan_material_manifest_commit": True,
        "shared_registries_modified_by_runner": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(comparison, output / "MODEL_COMPARISON.json")
    atomic_json(robustness, output / "DEV_ROBUSTNESS_REPORT.json")
    atomic_json(transfer, output / "SOURCE_TRANSFER_REPORT.json")
    atomic_json(diagnostic, output / "V72_CACHE_ADOPTION_REPORT.json")
    atomic_json(direction, output / "V72_DIRECTION_IMMUTABILITY_AUDIT.json")
    return {
        "status": "CACHE_ADOPTED_RESEARCH_FAIL",
        "output": str(output),
        "retraining_performed": False,
        "runner_files_written": [
            "MODEL_COMPARISON.json",
            "DEV_ROBUSTNESS_REPORT.json",
            "SOURCE_TRANSFER_REPORT.json",
            "V72_CACHE_ADOPTION_REPORT.json",
            "V72_DIRECTION_IMMUTABILITY_AUDIT.json",
        ],
        "standalone_manifest_sha256": cache_audit["standalone_manifest_sha256"],
        "selected_canonical_gate": selected["research_gate"],
        "material_gate": selected["material_gate"],
    }


def load_authorized_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    v69 = pd.read_csv(
        V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(v69) == EXPECTED_V69_ROWS, f"V69 OOF row count changed: {len(v69)}")
    require(len(dev) == EXPECTED_V36_ROWS, f"V36 DEV row count changed: {len(dev)}")
    require(v69.event_id.is_unique and dev.event_id.is_unique, "event_id uniqueness failed")
    require(set(v69.event_id).issubset(set(dev.event_id)), "V69 OOF is not a V36 DEV subset")
    aligned = dev.set_index("event_id").loc[v69.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), v69.y.to_numpy(int)), "V36/V69 labels differ")
    require(
        np.allclose(aligned.fwd_ret_30m, v69.fwd_ret_30m, rtol=0.0, atol=1e-15),
        "V36/V69 return outcomes differ",
    )
    require(aligned.source_family.astype(str).equals(v69.source_family.astype(str)), "source family differs")
    merged = v69.copy()
    for column in ("source", "event_type", "headline", "body", *STATE_COLUMNS):
        merged[column] = aligned[column].to_numpy()
    # The historical V69 OOF preserved an anomalous text-cluster serialization
    # for most US_NEWS rows.  Reconstruct the duplicate key only from the
    # immutable V36 headline/body, as V65 did, before any temporal split.
    merged["text_cluster_id"] = merged.apply(v37.normalized_cluster, axis=1)
    merged["cluster_weight"] = (
        1.0 / merged.groupby("text_cluster_id").event_id.transform("size")
    )
    merged["event_time_utc"] = pd.to_datetime(merged.event_time_utc, utc=True)
    merged["high_conf"] = bool_series(merged.high_conf)
    merged["direction_prediction"] = merged.prob.to_numpy(float) >= 0.5
    merged["correct_target"] = merged.direction_prediction.eq(merged.y.astype(bool)).astype(int)
    merged["material_target"] = (
        merged.fwd_ret_30m.abs().to_numpy(float) >= MATERIAL_MOVE_ABS_RETURN
    ).astype(int)
    merged["signed_net"] = (
        np.where(merged.direction_prediction, 1.0, -1.0)
        * merged.fwd_ret_30m.to_numpy(float) - COST
    )
    us_news = merged.loc[merged.source_family.eq("US_NEWS")].copy()
    us_news = us_news.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    require(len(us_news) == EXPECTED_US_NEWS_ROWS, f"US_NEWS count changed: {len(us_news)}")
    require(us_news.market.eq("US").all(), "US_NEWS includes a non-US row")
    require(us_news.correct_target.nunique() == 2, "correctness target has one class")
    require(us_news.material_target.nunique() == 2, "material target has one class")
    return v69, us_news


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: strict 35m embargo violated")
    overlap = set(train.text_cluster_id.astype(str)) & set(valid.text_cluster_id.astype(str))
    require(not overlap, f"{label}: text cluster leakage")
    return {
        "train_n": len(train),
        "valid_n": len(valid),
        "train_end": train_end,
        "valid_start": valid_start,
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "text_cluster_overlap_n": 0,
    }


def chronological_outer_folds(frame: pd.DataFrame, smoke: bool) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    ordered = frame.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    blocks = np.array_split(np.arange(len(ordered)), OUTER_BLOCKS)
    folds: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    for fold in range(1, OUTER_BLOCKS):
        valid = ordered.iloc[blocks[fold]].copy()
        boundary = valid.event_time_utc.min()
        train = ordered.loc[ordered.event_time_utc < boundary - EMBARGO].copy()
        train = train.loc[~train.text_cluster_id.astype(str).isin(set(valid.text_cluster_id.astype(str)))].copy()
        require(len(train) >= 120 and len(valid) >= 100, f"outer fold {fold} is too small")
        require(train.correct_target.nunique() == 2 and train.material_target.nunique() == 2,
                f"outer fold {fold} train target classes insufficient")
        folds.append((train, valid, fold))
    require(len(folds) == OUTER_BLOCKS - 1, "eligible outer fold count changed")
    return folds[:1] if smoke else folds


def inner_past_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True)
    split = max(90, int(INNER_TRAIN_FRACTION * len(ordered)))
    valid = ordered.iloc[split:].copy()
    boundary = valid.event_time_utc.min()
    base = ordered.loc[ordered.event_time_utc < boundary - EMBARGO].copy()
    base = base.loc[~base.text_cluster_id.astype(str).isin(set(valid.text_cluster_id.astype(str)))].copy()
    require(len(base) >= 70 and len(valid) >= 30, "inner-past split is too small")
    require(base.correct_target.nunique() == 2 and base.material_target.nunique() == 2,
            "inner-past train target classes insufficient")
    return base, valid


def smoothed_encoding(
    train: pd.DataFrame,
    target: pd.DataFrame,
    column: str,
    label: str,
    weight: str,
    smoothing: float = 12.0,
) -> np.ndarray:
    global_rate = float(np.average(train[label], weights=train[weight]))
    keys = train[column].fillna("<MISSING>").astype(str)
    work = pd.DataFrame({"key": keys, "label": train[label], "weight": train[weight]})
    work["weighted_label"] = work.label * work.weight
    stats = work.groupby("key", sort=False).agg(
        numerator=("weighted_label", "sum"), denominator=("weight", "sum")
    )
    encoded = (stats.numerator + smoothing * global_rate) / (stats.denominator + smoothing)
    return (
        target[column].fillna("<MISSING>").astype(str).map(encoded).fillna(global_rate).to_numpy(float)
    )


def engineered_features(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=target.index)
    base_up = target.prob.to_numpy(float) >= 0.5
    agreements = []
    margins = []
    aligned_probabilities = []
    reliability = []
    weights = train.cluster_weight.to_numpy(float)
    for column in EXPERT_PROBABILITY_COLUMNS:
        train_probability = pd.to_numeric(train[column], errors="coerce").to_numpy(float)
        target_probability = pd.to_numeric(target[column], errors="coerce").to_numpy(float)
        train_direction = train_probability >= 0.5
        target_direction = target_probability >= 0.5
        rel = float(np.average(train_direction == train.y.to_numpy(int).astype(bool), weights=weights))
        agree = (target_direction == base_up).astype(float)
        margin = 2.0 * np.abs(target_probability - 0.5)
        aligned = np.where(base_up, target_probability, 1.0 - target_probability)
        output[f"expert_probability__{column}"] = target_probability
        output[f"expert_margin__{column}"] = margin
        output[f"expert_agrees_base__{column}"] = agree
        output[f"expert_aligned_probability__{column}"] = aligned
        output[f"train_reliability__{column}"] = rel
        agreements.append(agree)
        margins.append(margin)
        aligned_probabilities.append(aligned)
        reliability.append(rel)
    agreement_matrix = np.column_stack(agreements)
    margin_matrix = np.column_stack(margins)
    aligned_matrix = np.column_stack(aligned_probabilities)
    reliability_vector = np.asarray(reliability, float)
    signed_agreement = 2.0 * agreement_matrix - 1.0
    reliability_weight = margin_matrix * reliability_vector[None, :]
    denominator = np.maximum(reliability_weight.sum(axis=1), 1e-12)
    output["expert_agreement_fraction"] = agreement_matrix.mean(axis=1)
    output["expert_disagreement"] = aligned_matrix.std(axis=1)
    output["expert_aligned_mean"] = aligned_matrix.mean(axis=1)
    output["expert_aligned_min"] = aligned_matrix.min(axis=1)
    output["expert_margin_mean"] = margin_matrix.mean(axis=1)
    output["reliability_weighted_agreement"] = (
        (signed_agreement * reliability_weight).sum(axis=1) / denominator
    )
    for column in EXPERT_CONFIDENCE_COLUMNS:
        output[f"frozen_confidence__{column}"] = pd.to_numeric(target[column], errors="coerce").to_numpy(float)
    output["direction_margin"] = np.abs(target.prob.to_numpy(float) - 0.5)
    output["predicted_up"] = base_up.astype(float)
    for category in ("source", "event_type", "ticker"):
        output[f"reliability_correct__{category}"] = smoothed_encoding(
            train, target, category, "correct_target", "cluster_weight"
        )
        output[f"reliability_material__{category}"] = smoothed_encoding(
            train, target, category, "material_target", "cluster_weight"
        )
    for column in STATE_COLUMNS:
        output[f"state__{column}"] = pd.to_numeric(target[column], errors="coerce").to_numpy(float)
    timestamp = pd.to_datetime(target.event_time_utc, utc=True)
    output["time_year"] = timestamp.dt.year.to_numpy(float)
    output["time_month_sin"] = np.sin(2.0 * np.pi * timestamp.dt.month.to_numpy(float) / 12.0)
    output["time_month_cos"] = np.cos(2.0 * np.pi * timestamp.dt.month.to_numpy(float) / 12.0)
    return output.replace([np.inf, -np.inf], np.nan)


def fit_dual_probabilities(
    train: pd.DataFrame, target: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_train = engineered_features(train, train)
    x_target = engineered_features(train, target)

    def fit(label: str, offset: int) -> tuple[np.ndarray, dict[str, Any]]:
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.25, solver="liblinear", class_weight="balanced",
                max_iter=1000, random_state=seed + offset,
            )),
        ])
        model.fit(x_train, train[label].to_numpy(int), model__sample_weight=train.cluster_weight.to_numpy(float))
        probability = np.clip(model.predict_proba(x_target)[:, 1], 1e-6, 1.0 - 1e-6)
        return probability, {
            "target": label,
            "model": "imputed_scaled_l2_logistic",
            "C": 0.25,
            "class_weight": "balanced",
            "feature_n": x_train.shape[1],
            "train_positive_rate": float(train[label].mean()),
        }

    p_correct, correct_audit = fit("correct_target", 0)
    p_material, material_audit = fit("material_target", 1)
    return p_correct, p_material, {
        "correctness_model": correct_audit,
        "material_move_model": material_audit,
        "models_are_separate": True,
        "shared_target_outcomes": False,
    }


def mask_metrics(frame: pd.DataFrame, mask: np.ndarray, score: np.ndarray | None = None) -> dict[str, Any]:
    high = np.asarray(mask, bool)
    correct = frame.correct_target.to_numpy(int)
    net = frame.signed_net.to_numpy(float)
    return {
        "n": len(frame),
        "selected_n": int(high.sum()),
        "coverage": float(high.mean()),
        "accuracy": float(correct[high].mean()) if high.any() else None,
        "mean_signed_net": float(net[high].mean()) if high.any() else None,
        "median_signed_net": float(np.median(net[high])) if high.any() else None,
        "positive_net_rate": float((net[high] > 0.0).mean()) if high.any() else None,
        "correctness_auc": safe_auc(correct, np.asarray(score, float)) if score is not None else None,
    }


def choose_inner_policy(
    frame: pd.DataFrame, p_correct: np.ndarray, p_material: np.ndarray
) -> dict[str, Any]:
    rows = []
    baseline = mask_metrics(frame, frame.high_conf.to_numpy(bool), frame.confidence_signal.to_numpy(float))
    for c_quantile in C_QUANTILES:
        c_cutoff = float(np.quantile(p_correct, c_quantile))
        for m_quantile in M_QUANTILES:
            m_cutoff = float(np.quantile(p_material, m_quantile))
            high = (p_correct >= c_cutoff) & (p_material >= m_cutoff)
            metrics = mask_metrics(frame, high, np.sqrt(p_correct * p_material))
            count = metrics["selected_n"]
            if count:
                selected_net = frame.loc[high, "signed_net"].to_numpy(float)
                net_lcb = float(selected_net.mean() - 0.5 * selected_net.std(ddof=1) / math.sqrt(max(1, count)))
            else:
                net_lcb = -math.inf
            eligible = bool(
                count >= 12
                and MIN_POLICY_COVERAGE <= metrics["coverage"] <= MAX_POLICY_COVERAGE
            )
            accuracy_delta = (
                metrics["accuracy"] - baseline["accuracy"]
                if metrics["accuracy"] is not None and baseline["accuracy"] is not None else -1.0
            )
            net_delta = (
                metrics["mean_signed_net"] - baseline["mean_signed_net"]
                if metrics["mean_signed_net"] is not None and baseline["mean_signed_net"] is not None else -1.0
            )
            rows.append({
                "c_quantile": c_quantile,
                "m_quantile": m_quantile,
                "c_cutoff": c_cutoff,
                "m_cutoff": m_cutoff,
                "eligible": eligible,
                "net_half_se_lcb": net_lcb,
                "accuracy_delta_vs_v69": accuracy_delta,
                "net_delta_vs_v69": net_delta,
                "metrics": metrics,
            })
    eligible_rows = [row for row in rows if row["eligible"]]
    require(eligible_rows, "inner coarse policy grid has no coverage-eligible policy")
    eligible_rows.sort(
        key=lambda row: (
            row["net_delta_vs_v69"] >= 0.0,
            row["accuracy_delta_vs_v69"] >= 0.0,
            row["net_half_se_lcb"],
            row["metrics"]["accuracy"],
            -abs(row["metrics"]["coverage"] - 0.15),
            row["c_quantile"] + row["m_quantile"],
        ),
        reverse=True,
    )
    return {
        "selected": eligible_rows[0],
        "candidates": rows,
        "baseline": baseline,
        "selection_uses_inner_labels_only": True,
    }


def run_nested(frame: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs = []
    audits = []
    for train, valid, fold in chronological_outer_folds(frame, smoke):
        inner_train, inner_valid = inner_past_split(train)
        inner_audit = chronology(inner_train, inner_valid, f"V72 inner fold {fold}")
        outer_audit = chronology(train, valid, f"V72 outer fold {fold}")
        inner_correct, inner_material, inner_models = fit_dual_probabilities(
            inner_train, inner_valid, SEED + fold * 10
        )
        policy = choose_inner_policy(inner_valid, inner_correct, inner_material)
        outer_correct, outer_material, outer_models = fit_dual_probabilities(
            train, valid, SEED + 1000 + fold * 10
        )
        selected = policy["selected"]
        high = (
            (outer_correct >= float(selected["c_cutoff"]))
            & (outer_material >= float(selected["m_cutoff"]))
        )
        output = valid[[
            "event_id", "event_group_id", "event_time_utc", "ticker", "source",
            "event_type", "source_family", "fold", "y", "fwd_ret_30m", "prob",
            "confidence_signal", "high_conf", "correct_target", "material_target",
            "signed_net", "cluster_weight", "text_cluster_id",
        ]].copy()
        output["meta_outer_fold"] = fold
        output["p_correct"] = outer_correct
        output["p_material_move"] = outer_material
        output["joint_confidence"] = np.sqrt(outer_correct * outer_material)
        output["dual_gate_high"] = high
        output["c_cutoff"] = float(selected["c_cutoff"])
        output["m_cutoff"] = float(selected["m_cutoff"])
        outputs.append(output)
        candidate_metrics = mask_metrics(output, high, output.joint_confidence.to_numpy(float))
        baseline_metrics = mask_metrics(
            output, output.high_conf.to_numpy(bool), output.confidence_signal.to_numpy(float)
        )
        audits.append({
            "outer_fold": fold,
            "inner_chronology": inner_audit,
            "outer_chronology": outer_audit,
            "inner_models": inner_models,
            "outer_models": outer_models,
            "inner_policy": policy,
            "outer_candidate": candidate_metrics,
            "outer_v69_baseline": baseline_metrics,
            "outer_labels_used_for_policy_selection": False,
            "direction_probability_changed": False,
        })
        print(
            f"[V72 DUAL] fold={fold} train={len(train)} valid={len(valid)} "
            f"c={selected['c_cutoff']:.4f} m={selected['m_cutoff']:.4f} "
            f"outer_selected={int(high.sum())}",
            flush=True,
        )
    evidence = pd.concat(outputs, ignore_index=True)
    return evidence, audits


def bootstrap_delta(frame: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = frame.copy().reset_index(drop=True)
    work["event_day"] = pd.to_datetime(work.event_time_utc, utc=True).dt.date
    groups = {day: np.asarray(index, int) for day, index in work.groupby("event_day").indices.items()}
    days = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(SEED + 99)
    candidate_net = []
    baseline_net = []
    net_delta = []
    accuracy_delta = []
    for _ in range(draws):
        positions = np.concatenate([groups[day] for day in rng.choice(days, len(days), replace=True)])
        sample = work.iloc[positions]
        candidate = sample.dual_gate_high.to_numpy(bool)
        baseline = sample.high_conf.to_numpy(bool)
        if not candidate.any() or not baseline.any():
            continue
        candidate_value = float(sample.loc[candidate, "signed_net"].mean())
        baseline_value = float(sample.loc[baseline, "signed_net"].mean())
        candidate_net.append(candidate_value)
        baseline_net.append(baseline_value)
        net_delta.append(candidate_value - baseline_value)
        accuracy_delta.append(
            float(sample.loc[candidate, "correct_target"].mean() - sample.loc[baseline, "correct_target"].mean())
        )

    def interval(values: list[float]) -> dict[str, Any]:
        require(values, "empty bootstrap distribution")
        return {
            "draws": len(values),
            "mean": float(np.mean(values)),
            "lower95": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "upper95": float(np.quantile(values, 0.975)),
            "probability_gt_zero": float(np.mean(np.asarray(values) > 0.0)),
        }

    return {
        "method": "paired event-day block bootstrap over frozen outer-OOF policies",
        "candidate_net": interval(candidate_net),
        "v69_baseline_net": interval(baseline_net),
        "net_delta": interval(net_delta),
        "accuracy_delta": interval(accuracy_delta),
    }


def tail_economics(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    values = frame.loc[np.asarray(mask, bool), "signed_net"].to_numpy(float)
    require(len(values) >= 10, "too few selected rows for tail robustness")
    ordered = np.sort(values)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "winsorized_1_99_mean": float(np.clip(values, *np.quantile(values, [0.01, 0.99])).mean()),
        "top_1_removed_mean": float(ordered[:-1].mean()),
        "top_5_removed_mean": float(ordered[:-5].mean()),
    }


def evaluate(evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    candidate = mask_metrics(
        evidence, evidence.dual_gate_high.to_numpy(bool), evidence.joint_confidence.to_numpy(float)
    )
    baseline = mask_metrics(
        evidence, evidence.high_conf.to_numpy(bool), evidence.confidence_signal.to_numpy(float)
    )
    p_correct_auc = safe_auc(evidence.correct_target, evidence.p_correct)
    p_material_auc = safe_auc(evidence.material_target, evidence.p_material_move)
    candidate_tail = tail_economics(evidence, evidence.dual_gate_high.to_numpy(bool))
    bootstrap = bootstrap_delta(evidence, draws)
    fold_rows = []
    for fold, part in evidence.groupby("meta_outer_fold"):
        candidate_fold = mask_metrics(part, part.dual_gate_high.to_numpy(bool), part.joint_confidence)
        baseline_fold = mask_metrics(part, part.high_conf.to_numpy(bool), part.confidence_signal)
        fold_rows.append({
            "fold": int(fold),
            "candidate": candidate_fold,
            "v69_baseline": baseline_fold,
            "net_delta": (
                candidate_fold["mean_signed_net"] - baseline_fold["mean_signed_net"]
                if candidate_fold["mean_signed_net"] is not None and baseline_fold["mean_signed_net"] is not None
                else None
            ),
        })
    positive_net_fold_fraction = float(np.mean([
        row["net_delta"] is not None and row["net_delta"] > 0.0 for row in fold_rows
    ]))
    checks = {
        "p_correct_auc_ge_v69_confidence_auc_plus_0_005": (
            p_correct_auc is not None and baseline["correctness_auc"] is not None
            and p_correct_auc >= baseline["correctness_auc"] + 0.005
        ),
        "p_material_move_auc_ge_0_520": p_material_auc is not None and p_material_auc >= 0.520,
        "hc_accuracy_delta_ge_0_020": (
            candidate["accuracy"] is not None and baseline["accuracy"] is not None
            and candidate["accuracy"] >= baseline["accuracy"] + 0.020
        ),
        "hc_mean_signed_net_gt_0": candidate["mean_signed_net"] is not None and candidate["mean_signed_net"] > 0.0,
        "hc_mean_signed_net_gt_v69": (
            candidate["mean_signed_net"] is not None and baseline["mean_signed_net"] is not None
            and candidate["mean_signed_net"] > baseline["mean_signed_net"]
        ),
        "bootstrap_candidate_net_lower95_gt_0": bootstrap["candidate_net"]["lower95"] > 0.0,
        "bootstrap_net_delta_probability_gt_zero_ge_0_85": (
            bootstrap["net_delta"]["probability_gt_zero"] >= 0.85
        ),
        "positive_net_delta_fold_fraction_ge_0_60": positive_net_fold_fraction >= 0.60,
        "coverage_between_0_07_and_0_30": MIN_POLICY_COVERAGE <= candidate["coverage"] <= MAX_POLICY_COVERAGE,
        "winsorized_net_gt_0": candidate_tail["winsorized_1_99_mean"] > 0.0,
        "top_5_removed_net_gt_0": candidate_tail["top_5_removed_mean"] > 0.0,
        "separate_correctness_and_material_models": all(
            audit["outer_models"]["models_are_separate"] for audit in audits
        ),
        "strict_chronology_and_embargo": all(
            audit["inner_chronology"]["text_cluster_overlap_n"] == 0
            and audit["outer_chronology"]["text_cluster_overlap_n"] == 0
            for audit in audits
        ),
        "outer_labels_not_used_for_selection": all(
            not audit["outer_labels_used_for_policy_selection"] for audit in audits
        ),
    }
    return {
        "candidate": candidate,
        "v69_baseline": baseline,
        "p_correct_correctness_auc": p_correct_auc,
        "p_material_move_auc": p_material_auc,
        "candidate_tail": candidate_tail,
        "bootstrap": bootstrap,
        "folds": fold_rows,
        "positive_net_delta_fold_fraction": positive_net_fold_fraction,
        "material_checks": checks,
        "material_pass": bool(all(checks.values())),
        "passed": int(sum(checks.values())),
        "total": len(checks),
    }


def fail_closed_frame(v69: pd.DataFrame, evidence: pd.DataFrame, material_pass: bool) -> pd.DataFrame:
    final = v69.copy()
    probability_before = final.prob.to_numpy(float).copy()
    direction_before = probability_before >= 0.5
    confidence_before = final.confidence_signal.to_numpy(float).copy()
    high_before = bool_series(final.high_conf).to_numpy(bool)
    if material_pass:
        lookup = evidence.set_index("event_id")
        eligible = final.event_id.isin(lookup.index) & final.source_family.eq("US_NEWS")
        ids = final.loc[eligible, "event_id"]
        final.loc[eligible, "confidence_signal"] = lookup.loc[ids, "joint_confidence"].to_numpy(float)
        final.loc[eligible, "high_conf"] = lookup.loc[ids, "dual_gate_high"].to_numpy(bool)
        final.loc[eligible, "family"] = "V72_US_NEWS_DUAL_CORRECTNESS_OPPORTUNITY_GATE"
    require(np.array_equal(final.prob.to_numpy(float), probability_before), "V72 changed V69 probability")
    require(np.array_equal(final.prob.to_numpy(float) >= 0.5, direction_before), "V72 changed V69 direction")
    if not material_pass:
        require(final.equals(v69), "V72 fail-closed frame is not an exact V69 no-op")
        require(np.array_equal(final.confidence_signal.to_numpy(float), confidence_before), "fallback confidence changed")
        require(np.array_equal(bool_series(final.high_conf).to_numpy(bool), high_before), "fallback high mask changed")
    return final


def direction_audit(v69: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    before_probability = v69.prob.to_numpy(float)
    after_probability = final.prob.to_numpy(float)
    before_direction = before_probability >= 0.5
    after_direction = after_probability >= 0.5
    non_news = v69.source_family.ne("US_NEWS").to_numpy(bool)
    return {
        "probability_mismatch_n": int(np.sum(before_probability != after_probability)),
        "direction_mismatch_n": int(np.sum(before_direction != after_direction)),
        "probability_sha256_before": array_sha256(before_probability),
        "probability_sha256_after": array_sha256(after_probability),
        "direction_sha256_before": array_sha256(before_direction),
        "direction_sha256_after": array_sha256(after_direction),
        "non_us_news_confidence_mismatch_n": int(np.sum(
            v69.loc[non_news, "confidence_signal"].to_numpy(float)
            != final.loc[non_news, "confidence_signal"].to_numpy(float)
        )),
        "non_us_news_high_conf_mismatch_n": int(np.sum(
            bool_series(v69.loc[non_news, "high_conf"]).to_numpy(bool)
            != bool_series(final.loc[non_news, "high_conf"]).to_numpy(bool)
        )),
        "exact_probability_unchanged": bool(np.array_equal(before_probability, after_probability)),
        "exact_direction_unchanged": bool(np.array_equal(before_direction, after_direction)),
    }


def write_draft_outputs(
    out: Path,
    authority: dict[str, Any],
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    final: pd.DataFrame,
    canonical: dict[str, Any],
    direction: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    status = "MATERIAL_PASS_DRAFT" if evaluation["material_pass"] else "MATERIAL_FAIL_EXACT_V69_FALLBACK"
    report = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "noncanonical_draft": True,
        "smoke_test": smoke,
        "authority_audit": authority,
        "contract": {
            "direction_authority": "committed V69, immutable",
            "correctness_target": "I[(V69 outer-OOF direction >= 0.5) == y]",
            "material_move_target": f"I[abs(fwd_ret_30m) >= {MATERIAL_MOVE_ABS_RETURN}]",
            "eligibility": "P(correct)>=c AND P(material_move)>=m",
            "policy_space": {
                "correctness_quantiles": C_QUANTILES,
                "material_quantiles": M_QUANTILES,
                "coverage_bounds": [MIN_POLICY_COVERAGE, MAX_POLICY_COVERAGE],
            },
            "selection": "coarse inner-past only; no outer outcome selection",
            "embargo_minutes": EMBARGO.total_seconds() / 60.0,
            "fallback": "exact committed V69 frame",
        },
        "evaluation": evaluation,
        "direction_immutability": direction,
        "canonical_controller_gate_recomputed_from_raw_metrics": canonical["research_gate"],
        "canonical_selected_metrics": canonical["metrics"],
        "nested_fold_audits": audits,
        "seal_authorized": False,
        "controller_commit_created": False,
        "registries_modified": False,
    }
    atomic_json(report, out / "V72_STANDALONE_REPORT.json")
    atomic_json(direction, out / "V72_DIRECTION_IMMUTABILITY_AUDIT.json")
    atomic_csv(evidence, out / "V72_DUAL_GATE_OUTER_OOF.csv.gz")
    atomic_csv(final, out / "V72_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    run_status = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "material_pass": evaluation["material_pass"],
        "phase": "DRAFT_STANDALONE_RESEARCH",
        "canonical_commit": False,
        "seal_state": "UNOPENED",
    }
    atomic_json(run_status, out / "V72_RUN_STATUS.json")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "V72_ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "noncanonical_draft": True,
        "authority_experiment_id": authority["experiment_id"],
        "files": files,
    }
    atomic_json(manifest, out / "V72_ARTIFACT_MANIFEST.json")
    return {
        "output": str(out),
        "files": len(files) + 1,
        "manifest_sha256": sha256(out / "V72_ARTIFACT_MANIFEST.json"),
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    authority = verify_v69_authority()
    if args.audit_only:
        print(json.dumps(json_clean({"status": "PASS", **authority}), indent=2))
        return
    if CONTROLLER_OUTPUT:
        require(args.output is None, "--output cannot be combined with MARKET_BIO_VERSION_OUTPUT")
        result = adopt_standalone_cache(Path(CONTROLLER_OUTPUT), authority)
        print(json.dumps(json_clean(result), indent=2))
        return
    v69, us_news = load_authorized_frame()
    evidence, audits = run_nested(us_news, args.smoke_test)
    evaluation = evaluate(
        evidence, audits, 250 if args.smoke_test else BOOTSTRAP_DRAWS
    )
    final = fail_closed_frame(v69, evidence, evaluation["material_pass"])
    direction = direction_audit(v69, final)
    require(direction["exact_probability_unchanged"], "V72 direction probability audit failed")
    require(direction["exact_direction_unchanged"], "V72 direction prediction audit failed")
    require(direction["non_us_news_confidence_mismatch_n"] == 0, "V72 changed non-US_NEWS confidence")
    require(direction["non_us_news_high_conf_mismatch_n"] == 0, "V72 changed non-US_NEWS mask")
    canonical = v44.summarize(final)
    out = args.output or (DEFAULT_OUT.with_name(DEFAULT_OUT.name + "_SMOKE") if args.smoke_test else DEFAULT_OUT)
    write_result = write_draft_outputs(
        out, authority, evidence, audits, evaluation, final, canonical, direction, args.smoke_test
    )
    print(json.dumps(json_clean({
        **write_result,
        "eligible_outer_oof_n": len(evidence),
        "candidate": evaluation["candidate"],
        "v69_baseline": evaluation["v69_baseline"],
        "p_correct_auc": evaluation["p_correct_correctness_auc"],
        "p_material_move_auc": evaluation["p_material_move_auc"],
        "material_gate": {
            "passed": evaluation["passed"],
            "total": evaluation["total"],
            "material_pass": evaluation["material_pass"],
        },
        "canonical_gate": canonical["research_gate"],
        "direction_immutability": direction,
    }), indent=2))


if __name__ == "__main__":
    main()
