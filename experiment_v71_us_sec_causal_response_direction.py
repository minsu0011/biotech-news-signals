"""V71 material US_SEC direction challenger: causal response priors.

The experiment is deliberately confined to the immutable V36 DEV contract.
It never imports a DEV-extension, research-seal, final-meta, or role file.
V69 remains the fail-closed champion.  Only US_SEC direction probabilities in
outer folds 2--4 may change in the diagnostic challenger; V69 confidence and
high-confidence membership remain exact.

Material new information is an expanding, past-only response prior for issuer,
SEC form, event taxonomy, form/event, issuer/event, and detected SEC semantic
taxonomy.  The prior is combined with structured SEC and pre-event context in
a chronological nested-OOF model.  Inner-past rows alone select one of three
predeclared information blocks and one of three coarse blend architectures.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import autonomous_v37plus as controller
import bio_news_30m_v3 as app
import experiment_v44_microstructure as v44
import experiment_v59_sec_structured_v1 as v59
import runtime_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
V69_DIR = ROOT / "output_V69"
DEV_PATH = DATA / "dev_contract_v36_labeled.csv.gz"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
V69_COMMIT_PATH = V69_DIR / "COMMIT.json"
V69_MANIFEST_PATH = V69_DIR / "ARTIFACT_MANIFEST.json"
V69_ROBUSTNESS_PATH = V69_DIR / "DEV_ROBUSTNESS_REPORT.json"
OUT = Path(
    os.environ.get(
        "MARKET_BIO_VERSION_OUTPUT",
        str(ROOT / "staging" / "V71_US_SEC_CAUSAL_RESPONSE_DIRECTION_V1"),
    )
)

VERSION = 71
HYPOTHESIS = "US_SEC_CAUSAL_RESPONSE_DIRECTION_V1"
EXPECTED_SHA256 = {
    DEV_PATH: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V69_PATH: "f532a01fba5633e8d549b04d45570bd087e7272a728a907b3633e2a9e71b5002",
}
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V69_COMMIT_SHA256 = "8e58efadc248661ef5b276d4e0727c3ab7b677d3716bba5bcc8b441497d6ec31"
EXPECTED_V69_MANIFEST_SHA256 = "10a3bd932a3a8a358cc707a7e5bfa4e8b4b8e51d20fff73a6083940cd4b3e836"
EXPECTED_DEV_ROWS = 9462
EXPECTED_OOF_ROWS = 7568
EXPECTED_US_SEC_OOF_ROWS = 5426
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
PRIOR_STRENGTH = 16.0
MODEL_C = 0.25
BLEND_WEIGHTS = (0.25, 0.50, 0.75)
SEED = 7101
BOOTSTRAP_DRAWS = 2000
COST = 0.002

# This is a deliberately compact causal market/sector-state block rather than
# a duplicate of every V36 numeric feature.
CONTEXT_COLUMNS = (
    "pre_ret_2", "pre_ret_5", "pre_ret_15", "pre_ret_30", "pre_ret_60",
    "volume_ratio_2_30", "volume_ratio_5_30", "range_10m", "range_30m",
    "close_position_30", "intraday_ret_open", "minutes_from_open", "minutes_to_close",
    "benchmark_ret_2", "benchmark_ret_5", "benchmark_ret_15", "benchmark_ret_30",
    "benchmark_ret_60", "excess_ret_5", "excess_ret_15", "excess_ret_30",
    "excess_ret_60", "event_positive_kw", "event_negative_kw", "event_financing_kw",
    "event_trial_kw", "event_regulatory_kw", "event_ma_kw", "event_earnings_kw",
)

SPECS = (
    {"name": "CAUSAL_PRIOR_ONLY", "structured": False, "context": False},
    {"name": "STRUCTURED_CAUSAL_PRIOR", "structured": True, "context": False},
    {"name": "STRUCTURED_CAUSAL_CONTEXT", "structured": True, "context": True},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    raw = (json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(raw, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if path.name.endswith(".gz"):
        with temporary.open("wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
                compressed.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary.write_bytes(raw)
    os.replace(temporary, path)


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    return float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None


def metric(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    y = frame.y.to_numpy(int)
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    return {
        "n": len(frame),
        "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(prediction == y)),
        "pred_up": float(prediction.mean()),
    }


def probability_blend(baseline: np.ndarray, challenger: np.ndarray, weight: float) -> np.ndarray:
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    challenger_logit = logit(np.clip(np.asarray(challenger, float), 1e-5, 1.0 - 1e-5))
    return expit((1.0 - float(weight)) * base_logit + float(weight) * challenger_logit)


def verify_and_load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hashes: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        require(path.is_file(), f"required V71 input missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"pinned V71 input changed: {path.name}")
        hashes[str(path.relative_to(ROOT))] = actual
    require(sha256(V69_COMMIT_PATH) == EXPECTED_V69_COMMIT_SHA256, "V69 atomic COMMIT changed")
    require(sha256(V69_MANIFEST_PATH) == EXPECTED_V69_MANIFEST_SHA256, "V69 artifact manifest changed")
    commit = json.loads(V69_COMMIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(V69_MANIFEST_PATH.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 COMMIT experiment mismatch")
    require(commit.get("manifest_sha256") == EXPECTED_V69_MANIFEST_SHA256, "V69 COMMIT/manifest binding mismatch")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest experiment mismatch")
    require(
        manifest.get("files", {}).get(V69_PATH.name, {}).get("sha256") == EXPECTED_SHA256[V69_PATH],
        "V69 selected OOF is not bound by its committed manifest",
    )
    require(
        manifest.get("files", {}).get(V69_ROBUSTNESS_PATH.name, {}).get("sha256") == sha256(V69_ROBUSTNESS_PATH),
        "V69 robustness report is not bound by its committed manifest",
    )
    data = pd.read_csv(DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    champion = pd.read_csv(V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str})
    require(len(data) == EXPECTED_DEV_ROWS, "immutable DEV row count changed")
    require(len(champion) == EXPECTED_OOF_ROWS, "V69 OOF row count changed")
    require(champion.event_id.is_unique, "V69 event_id is not unique")
    require(set(champion.event_id).issubset(set(data.event_id)), "V69 contains an event outside immutable DEV")
    require(int(champion.source_family.eq("US_SEC").sum()) == EXPECTED_US_SEC_OOF_ROWS, "US_SEC OOF count changed")
    data["event_time_utc"] = pd.to_datetime(data.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    data["headline"] = data.headline.fillna("").astype(str)
    data["body"] = data.body.fillna("").astype(str)
    data, recovery = v59.recover_bodies(data)
    sec = data.loc[data.source_family.eq("US_SEC")].copy()
    require(sec.event_id.is_unique and len(sec) == 7022, "immutable SEC training universe changed")
    require(sec.y.isin([0, 1]).all(), "invalid DEV direction label")
    recovered_manifest = sorted(recovery, key=lambda row: row["event_id"])
    audit = {
        "input_sha256": hashes,
        "authorized_inputs": [str(DEV_PATH.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT))],
        "opened_dev_only": True,
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "research_seal_pool_loaded": False,
        "final_meta_reserve_loaded": False,
        "seal_or_final_paths_present_in_runner": False,
        "dev_rows": len(data),
        "sec_training_rows": len(sec),
        "v69_oof_rows": len(champion),
        "v69_us_sec_oof_rows": int(champion.source_family.eq("US_SEC").sum()),
        "v69_atomic_commit": {
            "experiment_id": EXPECTED_V69_EXPERIMENT_ID,
            "commit_sha256": EXPECTED_V69_COMMIT_SHA256,
            "manifest_sha256": EXPECTED_V69_MANIFEST_SHA256,
            "selected_oof_manifest_bound": True,
            "robustness_report_manifest_bound": True,
        },
        "recovered_sec_text_files": len(recovered_manifest),
        "recovered_sec_text_manifest_sha256": hashlib.sha256(
            json.dumps(recovered_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return sec, champion, audit


def taxonomy_tokens(structure: dict[str, float]) -> tuple[str, ...]:
    tokens = sorted(
        key.split("=", 1)[1]
        for key, value in structure.items()
        if key.startswith("tax=") and float(value) > 0.0
    )
    return tuple(tokens or ["NONE"])


def prepare_sec(sec: pd.DataFrame) -> pd.DataFrame:
    work = sec.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True).copy()
    structures = [v59.structured_features(row) for _, row in work.iterrows()]
    work["sec_structure"] = structures
    work["taxonomy_tokens"] = [taxonomy_tokens(value) for value in structures]
    model_frame = app.model_frame(work)
    for column in CONTEXT_COLUMNS:
        work[column] = pd.to_numeric(model_frame[column], errors="coerce")
    return work


def single_keys(row: pd.Series) -> dict[str, str]:
    ticker = str(row.ticker)
    form = str(row.form or "UNKNOWN").upper()
    event = str(row.event_type or "UNKNOWN").upper()
    return {
        "issuer": ticker,
        "form": form,
        "event": event,
        "form_event": f"{form}|{event}",
        "issuer_event": f"{ticker}|{event}",
    }


def shrunk_prior(total: float, count: int, global_prior: float) -> float:
    return float((total + PRIOR_STRENGTH * global_prior) / (count + PRIOR_STRENGTH))


def prior_features(
    frame: pd.DataFrame,
    stats: dict[str, dict[str, list[float]]],
    global_prior: float,
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        values: dict[str, float] = {"prior_global": float(global_prior)}
        for family, key in single_keys(row).items():
            total, count = stats[family].get(key, [0.0, 0.0])
            prior = shrunk_prior(total, int(count), global_prior)
            values[f"prior_{family}"] = prior
            values[f"prior_{family}_delta"] = prior - global_prior
            values[f"prior_{family}_log_count"] = float(np.log1p(count))
        taxonomy_rows = [stats["taxonomy"].get(token, [0.0, 0.0]) for token in row.taxonomy_tokens]
        taxonomy_priors = [shrunk_prior(total, int(count), global_prior) for total, count in taxonomy_rows]
        taxonomy_count = sum(int(count) for _, count in taxonomy_rows)
        taxonomy_prior = float(np.mean(taxonomy_priors))
        values["prior_taxonomy"] = taxonomy_prior
        values["prior_taxonomy_delta"] = taxonomy_prior - global_prior
        values["prior_taxonomy_log_count"] = float(np.log1p(taxonomy_count))
        output.append(values)
    return output


def update_stats(stats: dict[str, dict[str, list[float]]], row: pd.Series) -> None:
    y = float(row.y)
    for family, key in single_keys(row).items():
        cell = stats[family][key]
        cell[0] += y
        cell[1] += 1.0
    for token in row.taxonomy_tokens:
        cell = stats["taxonomy"][str(token)]
        cell[0] += y
        cell[1] += 1.0


def causal_prior_blocks(train: pd.DataFrame, target: pd.DataFrame) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    ordered = train.sort_values(["event_time_utc", "event_id"], kind="stable")
    stats: dict[str, dict[str, list[float]]] = {
        family: defaultdict(lambda: [0.0, 0.0])
        for family in ("issuer", "form", "event", "form_event", "issuer_event", "taxonomy")
    }
    train_output: list[dict[str, float]] = []
    global_total = 0.0
    global_count = 0
    for _, row in ordered.iterrows():
        global_prior = (global_total + PRIOR_STRENGTH * 0.5) / (global_count + PRIOR_STRENGTH)
        train_output.extend(prior_features(row.to_frame().T, stats, float(global_prior)))
        update_stats(stats, row)
        global_total += float(row.y)
        global_count += 1
    final_global = (global_total + PRIOR_STRENGTH * 0.5) / (global_count + PRIOR_STRENGTH)
    target_output = prior_features(target, stats, float(final_global))
    # Return train features in the original train row order expected below.
    index_to_feature = dict(zip(ordered.index, train_output))
    return [index_to_feature[index] for index in train.index], target_output


def feature_dicts(
    frame: pd.DataFrame,
    priors: list[dict[str, float]],
    spec: dict[str, Any],
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        values = dict(priors[position])
        if spec["structured"]:
            values.update({f"sec::{key}": float(value) for key, value in row.sec_structure.items()})
        if spec["context"]:
            for column in CONTEXT_COLUMNS:
                value = finite(row[column])
                if value is not None:
                    values[f"context::{column}"] = float(np.clip(value, -20.0, 20.0))
        output.append(values)
    return output


class CausalResponseModel:
    def __init__(self, spec: dict[str, Any]):
        self.spec = dict(spec)
        self.pipeline = Pipeline(
            [
                ("vector", DictVectorizer(sparse=True, dtype=np.float64)),
                ("scale", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        C=MODEL_C,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=600,
                        random_state=SEED,
                    ),
                ),
            ]
        )

    def fit_predict(self, train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
        train_prior, target_prior = causal_prior_blocks(train, target)
        train_x = feature_dicts(train, train_prior, self.spec)
        target_x = feature_dicts(target, target_prior, self.spec)
        self.pipeline.fit(train_x, train.y.astype(int), model__sample_weight=np.ones(len(train), dtype=float))
        probability = self.pipeline.predict_proba(target_x)[:, 1]
        require(np.isfinite(probability).all(), "causal response model produced non-finite probability")
        return probability


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) > 0 and len(valid) > 0, f"{label}: empty split")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: 35-minute embargo failed")
    overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    require(not overlap, f"{label}: event-group overlap")
    return {
        "train_n": len(train),
        "valid_n": len(valid),
        "train_end": train_end.isoformat(),
        "valid_start": valid_start.isoformat(),
        "strict_35m_embargo": True,
        "event_group_overlap": 0,
    }


def inner_partition(sec: pd.DataFrame, champion_sec: pd.DataFrame, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible = champion_sec.loc[champion_sec.event_time_utc < outer_start - EMBARGO].copy()
    eligible = eligible.sort_values(["event_time_utc", "event_id"], kind="stable")
    require(len(eligible) >= 300, "insufficient prior OOF evidence for nested selection")
    boundary = max(100, int(math.floor(len(eligible) * (1.0 - INNER_VALID_FRACTION))))
    inner_valid_ids = set(eligible.iloc[boundary:].event_id)
    inner_valid = sec.loc[sec.event_id.isin(inner_valid_ids)].copy()
    inner_valid = inner_valid.sort_values(["event_time_utc", "event_id"], kind="stable")
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = sec.loc[sec.event_time_utc < inner_start - EMBARGO].copy()
    duplicate = set(inner_valid.event_group_id.astype(str))
    inner_train = inner_train.loc[~inner_train.event_group_id.astype(str).isin(duplicate)].copy()
    audit = chronology(inner_train, inner_valid, "inner")
    audit["inner_valid_fraction_target"] = INNER_VALID_FRACTION
    audit["inner_valid_is_prior_chronological_oof"] = True
    return inner_train, inner_valid, audit


def choose_inner(
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    champion_lookup: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    baseline = champion_lookup.set_index("event_id").loc[inner_valid.event_id, "prob"].to_numpy(float)
    baseline_metrics = metric(inner_valid, baseline)
    predictions: dict[str, np.ndarray] = {}
    trials: list[dict[str, Any]] = [
        {
            "name": "V69_NOOP",
            "spec": None,
            "challenger_weight": 0.0,
            "metrics": baseline_metrics,
            "auc_delta": 0.0,
            "balanced_accuracy_delta": 0.0,
            "eligible": True,
            "score": float(2.0 * baseline_metrics["auc"] + baseline_metrics["balanced_accuracy"]),
        }
    ]
    for spec in SPECS:
        causal_probability = CausalResponseModel(spec).fit_predict(inner_train, inner_valid)
        predictions[spec["name"]] = causal_probability
        for weight in BLEND_WEIGHTS:
            probability = probability_blend(baseline, causal_probability, weight)
            values = metric(inner_valid, probability)
            auc_delta = values["auc"] - baseline_metrics["auc"]
            ba_delta = values["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
            eligible = ba_delta >= -0.010
            trials.append(
                {
                    "name": f"{spec['name']}_W{weight:.2f}",
                    "spec": spec,
                    "challenger_weight": weight,
                    "metrics": values,
                    "auc_delta": auc_delta,
                    "balanced_accuracy_delta": ba_delta,
                    "eligible": eligible,
                    "score": float(2.0 * values["auc"] + values["balanced_accuracy"]),
                }
            )
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], -trial["challenger_weight"]),
    )
    return {
        "selection_rule": "maximize 2*AUC+BA on inner-past OOF, BA delta >= -0.010; tie -> AUC then smaller blend",
        "baseline": baseline_metrics,
        "selected": selected,
        "trials": trials,
        "outer_labels_used_for_selection": False,
    }, predictions


def run_nested(sec: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v71_applied"] = False
    diagnostic["v71_model"] = "V69_NOOP"
    diagnostic["v71_challenger_weight"] = 0.0
    champion_sec = champion.loc[champion.source_family.eq("US_SEC")].copy()
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold in (2, 3, 4):
        outer_champion = champion_sec.loc[champion_sec.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        outer_valid = sec.set_index("event_id").loc[outer_champion.event_id].reset_index()
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = sec.loc[sec.event_time_utc < outer_start - EMBARGO].copy()
        duplicate = set(outer_valid.event_group_id.astype(str))
        outer_train = outer_train.loc[~outer_train.event_group_id.astype(str).isin(duplicate)].copy()
        outer_audit = chronology(outer_train, outer_valid, f"outer fold {fold}")
        inner_train, inner_valid, inner_audit = inner_partition(sec, champion_sec, outer_start)
        policy, _ = choose_inner(inner_train, inner_valid, champion_sec)
        selected = policy["selected"]
        baseline_outer = outer_champion.prob.to_numpy(float)
        if selected["spec"] is None:
            outer_probability = baseline_outer.copy()
        else:
            causal_outer = CausalResponseModel(selected["spec"]).fit_predict(outer_train, outer_valid)
            outer_probability = probability_blend(
                baseline_outer, causal_outer, float(selected["challenger_weight"])
            )
        baseline_metrics = metric(outer_valid, baseline_outer)
        candidate_metrics = metric(outer_valid, outer_probability)
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, outer_probability))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v71_applied"] = selected["spec"] is not None
        diagnostic.loc[positions, "v71_model"] = selected["name"]
        diagnostic.loc[positions, "v71_challenger_weight"] = float(selected["challenger_weight"])
        evidence = outer_champion[
            ["event_id", "event_group_id", "event_time_utc", "fold", "y", "fwd_ret_30m", "high_conf"]
        ].copy()
        evidence["baseline_prob"] = baseline_outer
        evidence["candidate_prob"] = outer_probability
        evidence["baseline_direction"] = baseline_outer >= 0.5
        evidence["candidate_direction"] = outer_probability >= 0.5
        evidence["v71_model"] = selected["name"]
        evidence["v71_challenger_weight"] = float(selected["challenger_weight"])
        evidence_parts.append(evidence)
        audits.append(
            {
                "fold": fold,
                "outer_chronology": outer_audit,
                "inner_chronology": inner_audit,
                "policy": policy,
                "outer_baseline": baseline_metrics,
                "outer_candidate": candidate_metrics,
                "outer_auc_delta": candidate_metrics["auc"] - baseline_metrics["auc"],
                "outer_balanced_accuracy_delta": (
                    candidate_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
                ),
                "policy_locked_before_outer_evaluation": True,
                "outer_labels_used_for_selection": False,
            }
        )
        print(
            f"[V71] fold={fold} model={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} "
            f"outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
        if smoke:
            break
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "outer evidence repeats an event")
    return diagnostic, evidence, audits


def paired_date_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    work["date"] = pd.to_datetime(work.event_time_utc, utc=True).dt.strftime("%Y-%m-%d")
    blocks = [part for _, part in work.groupby(["fold", "date"], sort=True)]
    require(len(blocks) >= 10, "too few date blocks for paired bootstrap")
    generator = np.random.default_rng(SEED)
    auc_delta: list[float] = []
    ba_delta: list[float] = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, base)))
        ba_delta.append(
            float(
                balanced_accuracy_score(y, candidate >= 0.5)
                - balanced_accuracy_score(y, base >= 0.5)
            )
        )
    require(len(auc_delta) >= int(0.95 * draws), "bootstrap lost too many single-class samples")
    return {
        "method": "paired outer-fold/date-block bootstrap over fixed nested policies",
        "seed": SEED,
        "requested_draws": draws,
        "effective_draws": len(auc_delta),
        "date_blocks": len(blocks),
        "auc_delta_probability_gt_zero": float(np.mean(np.asarray(auc_delta) > 0.0)),
        "auc_delta_quantiles": {
            str(q): float(np.quantile(auc_delta, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
        "balanced_accuracy_delta_probability_gt_zero": float(np.mean(np.asarray(ba_delta) > 0.0)),
        "balanced_accuracy_delta_quantiles": {
            str(q): float(np.quantile(ba_delta, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)
        },
    }


def evaluate(
    champion: pd.DataFrame,
    diagnostic: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    draws: int,
) -> dict[str, Any]:
    # The baseline summary is part of V69's verified atomic commit.  Reusing
    # it avoids a redundant stochastic bootstrap while keeping the exact
    # controller-recomputed gate as the authority.
    baseline_report = json.loads(V69_ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    candidate_summary = v44.summarize(diagnostic)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(candidate_summary["research_gate"] == canonical_candidate, "V71/controller canonical gate mismatch")
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller canonical gate mismatch")
    outer_base = metric(evidence, evidence.baseline_prob.to_numpy(float))
    outer_candidate = metric(evidence, evidence.candidate_prob.to_numpy(float))
    fold_auc_deltas = [float(audit["outer_auc_delta"]) for audit in audits]
    bootstrap = paired_date_bootstrap(evidence, draws)
    base_source = baseline_summary["metrics"]["by_source_family"]["US_SEC"]
    candidate_source = candidate_summary["metrics"]["by_source_family"]["US_SEC"]
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    non_sec = champion.source_family.ne("US_SEC").to_numpy()
    confidence_exact = np.array_equal(
        champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)
    ) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool))
    material_checks = {
        "outer_evidence_auc_delta_gt_0_003": outer_candidate["auc"] - outer_base["auc"] > 0.003,
        "outer_evidence_balanced_accuracy_delta_gt_0": (
            outer_candidate["balanced_accuracy"] - outer_base["balanced_accuracy"] > 0.0
        ),
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(np.asarray(fold_auc_deltas) > 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta_probability_gt_zero"] >= 0.75,
        "full_us_sec_auc_delta_gt_0_002": candidate_source["auc"] - base_source["auc"] > 0.002,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": (
            candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001
        ),
        "hc_accuracy_delta_ge_minus_0_005": (
            candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005
        ),
        "hc_net_delta_ge_minus_0_0005": (
            candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005
        ),
        "v69_confidence_and_highconf_exact": confidence_exact,
        "non_us_sec_probabilities_exact": np.array_equal(
            champion.loc[non_sec, "prob"].to_numpy(float), diagnostic.loc[non_sec, "prob"].to_numpy(float)
        ),
        "strict_nested_chronology_all_folds": all(
            audit["outer_chronology"]["strict_35m_embargo"]
            and audit["inner_chronology"]["strict_35m_embargo"]
            and not audit["outer_labels_used_for_selection"]
            for audit in audits
        ),
    }
    material_pass = bool(all(material_checks.values()))
    selected = diagnostic if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    selected_canonical = controller.canonical_research_gate(selected_summary)
    require(selected_summary["research_gate"] == selected_canonical, "selected/controller gate mismatch")
    require(
        material_pass
        or (
            np.array_equal(selected.prob.to_numpy(float), champion.prob.to_numpy(float))
            and np.array_equal(selected.high_conf.to_numpy(bool), champion.high_conf.to_numpy(bool))
            and np.array_equal(selected.confidence_signal.to_numpy(float), champion.confidence_signal.to_numpy(float))
        ),
        "V71 fail-closed fallback is not exact V69",
    )
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_V69_NOOP",
        "material_pass": material_pass,
        "material_checks": material_checks,
        "outer_baseline": outer_base,
        "outer_candidate": outer_candidate,
        "outer_auc_delta": outer_candidate["auc"] - outer_base["auc"],
        "outer_balanced_accuracy_delta": (
            outer_candidate["balanced_accuracy"] - outer_base["balanced_accuracy"]
        ),
        "positive_outer_fold_auc_fraction": float(np.mean(np.asarray(fold_auc_deltas) > 0.0)),
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {
            "baseline": canonical_baseline,
            "candidate": canonical_candidate,
            "selected": selected_canonical,
            "reported_equals_controller_recomputed": True,
        },
        "direction_change": {
            "probability_changed_n": int(
                np.sum(champion.prob.to_numpy(float) != diagnostic.prob.to_numpy(float))
            ),
            "direction_changed_n": int(
                np.sum((champion.prob.to_numpy(float) >= 0.5) != (diagnostic.prob.to_numpy(float) >= 0.5))
            ),
            "baseline_probability_sha256": array_sha256(champion.prob.to_numpy(float)),
            "candidate_probability_sha256": array_sha256(diagnostic.prob.to_numpy(float)),
            "selected_probability_sha256": array_sha256(selected.prob.to_numpy(float)),
            "explicit_direction_challenger": True,
        },
        "fallback": {
            "activated": not material_pass,
            "policy": "exact V69 prob/confidence_signal/high_conf" if not material_pass else None,
            "exact_arrays_verified": bool(
                material_pass
                or (
                    np.array_equal(selected.prob.to_numpy(float), champion.prob.to_numpy(float))
                    and np.array_equal(selected.high_conf.to_numpy(bool), champion.high_conf.to_numpy(bool))
                    and np.array_equal(selected.confidence_signal.to_numpy(float), champion.confidence_signal.to_numpy(float))
                )
            ),
        },
        "diagnostic_frame": diagnostic,
        "selected_frame": selected,
    }


def build_reports(
    champion: pd.DataFrame,
    evidence: pd.DataFrame,
    audits: list[dict[str, Any]],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    selected_contract = dict(evaluation["selected_summary"])
    selected_contract["material_gate"] = {
        "contract": HYPOTHESIS,
        "checks": evaluation["material_checks"],
        "passed": int(sum(evaluation["material_checks"].values())),
        "total": len(evaluation["material_checks"]),
        "material_pass": evaluation["material_pass"],
    }
    validation = (
        "Immutable V36 DEV only; V69 fail-closed champion; outer folds 2-4; predeclared causal response, structured SEC, "
        "and pre-event context blocks; expanding past-only target priors; inner-past OOF architecture selection; strict "
        "35-minute embargo; outer outcomes never select a policy; V69 confidence/high_conf exact; controller recomputes "
        "the canonical 14 gates; no seal/final/role access."
    )
    comparison = {
        "version": "V71",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "material_change": "new causal issuer/form/event/taxonomy response priors for US_SEC direction",
        "models": {
            "V69_CHAMPION": evaluation["baseline_summary"],
            "V71_DIAGNOSTIC_CHALLENGER": evaluation["candidate_summary"],
            "V71_FAIL_CLOSED_SELECTED": evaluation["selected_summary"],
        },
        "evaluation": compact,
        "input_audit": input_audit,
        "validation": validation,
        "seal_authorized": False,
    }
    robustness = {
        "version": "V71",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "opened_dev_only": True,
        "seal_outcomes_loaded": False,
        "final_meta_outcomes_loaded": False,
        "selected": selected_contract,
        "candidate": evaluation["candidate_summary"],
        "baseline": evaluation["baseline_summary"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    transfer = {
        "version": "V71",
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "selected_oof_by_source": evaluation["selected_summary"]["metrics"]["by_source_family"],
        "selected_oof_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
        "scope": "US_SEC direction only; V69 confidence/high_conf frozen",
        "all_non_us_sec_probabilities_unchanged": True,
        "dev_extension_loaded": False,
        "seal_outcomes_loaded": False,
        "method": validation,
    }
    hypothesis_report = {
        "version": "V71",
        "hypothesis": HYPOTHESIS,
        "target_bottleneck": "US_SEC direction ranking/AUC",
        "material_information": {
            "causal_priors": ["issuer", "form", "event", "form_event", "issuer_event", "SEC taxonomy"],
            "prior_strength": PRIOR_STRENGTH,
            "structured_sec": True,
            "pre_event_context_columns": list(CONTEXT_COLUMNS),
            "raw_sign_flip": False,
            "future_or_current_target_used_in_prior": False,
        },
        "candidate_specs": list(SPECS),
        "blend_weights": list(BLEND_WEIGHTS),
        "nested_fold_audits": audits,
        "evaluation": compact,
        "input_audit": input_audit,
        "validation": validation,
    }
    gate_audit = {
        "version": "V71",
        "status": "MATCH",
        "reported": evaluation["selected_summary"]["research_gate"],
        "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        "material_gate": selected_contract["material_gate"],
    }
    fold_rows = []
    for audit in audits:
        selected = audit["policy"]["selected"]
        fold_rows.append(
            {
                "fold": audit["fold"],
                "outer_train_n": audit["outer_chronology"]["train_n"],
                "outer_valid_n": audit["outer_chronology"]["valid_n"],
                "inner_train_n": audit["inner_chronology"]["train_n"],
                "inner_valid_n": audit["inner_chronology"]["valid_n"],
                "selected_model": selected["name"],
                "selected_weight": selected["challenger_weight"],
                "inner_auc_delta": selected["auc_delta"],
                "inner_ba_delta": selected["balanced_accuracy_delta"],
                "outer_auc_delta": audit["outer_auc_delta"],
                "outer_ba_delta": audit["outer_balanced_accuracy_delta"],
                "strict_35m_embargo": True,
                "outer_labels_used_for_selection": False,
            }
        )
    json_reports = {
        "MODEL_COMPARISON.json": comparison,
        "DEV_ROBUSTNESS_REPORT.json": robustness,
        "SOURCE_TRANSFER_REPORT.json": transfer,
        "US_SEC_CAUSAL_RESPONSE_DIRECTION_REPORT.json": hypothesis_report,
        "BOOTSTRAP_DIRECTION_DELTA_REPORT.json": evaluation["bootstrap"],
        "CANONICAL_RESEARCH_GATE_AUDIT.json": gate_audit,
    }
    selected_columns = list(champion.columns)
    table_reports = {
        "V71_US_SEC_CAUSAL_FOLD_AUDIT.csv": pd.DataFrame(fold_rows),
        "V71_US_SEC_DIRECTION_EVIDENCE_OOF.csv.gz": evidence,
        "V71_DIAGNOSTIC_CHALLENGER_OOF.csv.gz": evaluation["diagnostic_frame"],
        "V71_FAIL_CLOSED_SELECTED_OOF.csv.gz": evaluation["selected_frame"][selected_columns],
    }
    return json_reports, table_reports


def write_outputs(
    json_reports: dict[str, Any],
    table_reports: dict[str, pd.DataFrame],
    evaluation: dict[str, Any],
    input_audit: dict[str, Any],
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in json_reports.items():
        atomic_json(payload, OUT / name)
    for name, frame in table_reports.items():
        atomic_csv(frame, OUT / name)
    material_spec = {
        "hypothesis": HYPOTHESIS,
        "prior_strength": PRIOR_STRENGTH,
        "model_c": MODEL_C,
        "specs": SPECS,
        "blend_weights": BLEND_WEIGHTS,
        "embargo_seconds": int(EMBARGO.total_seconds()),
        "validation": "nested chronological OOF",
    }
    experiment_id = hashlib.sha256(
        (
            "|".join(input_audit["input_sha256"].values())
            + "|"
            + json.dumps(material_spec, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    run_status = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "status": evaluation["status"],
        "phase": "ROBUST_SURVIVOR" if evaluation["selected_summary"]["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
        "canonical_gate_passed": evaluation["selected_summary"]["research_gate"]["passed"],
        "canonical_gate_total": evaluation["selected_summary"]["research_gate"]["total"],
        "material_pass": evaluation["material_pass"],
        "champion_changed": evaluation["material_pass"],
        "exit_code": 0,
        "completed_at": now(),
        "experiment_id": experiment_id,
        "seal_state": "UNOPENED",
        "seal_authorized": False,
        "final_status": "CONTINUE",
        "output_scope": str(OUT.relative_to(ROOT)) if OUT.is_relative_to(ROOT) else str(OUT),
    }
    atomic_json(run_status, OUT / "RUN_STATUS.json")
    files: dict[str, Any] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json", "COMMIT.json"}:
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"version": VERSION, "hypothesis": HYPOTHESIS, "experiment_id": experiment_id, "files": files}
    atomic_json(manifest, OUT / "ARTIFACT_MANIFEST.json")
    commit = {
        "version": VERSION,
        "hypothesis": HYPOTHESIS,
        "committed_at": now(),
        "experiment_id": experiment_id,
        "input_sha256": input_audit["input_sha256"],
        "material_fingerprint": hashlib.sha256(json.dumps(material_spec, sort_keys=True).encode("utf-8")).hexdigest(),
        "manifest_sha256": sha256(OUT / "ARTIFACT_MANIFEST.json"),
    }
    atomic_json(commit, OUT / "COMMIT.json")
    return {"experiment_id": experiment_id, "artifact_count": len(files) + 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    sec, champion, input_audit = verify_and_load()
    sec = prepare_sec(sec)
    if args.audit_only:
        print(json.dumps(clean({"status": "AUDIT_OK", "input_audit": input_audit, "output_written": False}), indent=2))
        return
    diagnostic, evidence, audits = run_nested(sec, champion, smoke=args.smoke_test)
    evaluation = evaluate(
        champion,
        diagnostic,
        evidence,
        audits,
        draws=250 if args.smoke_test else BOOTSTRAP_DRAWS,
    )
    summary = {
        "status": evaluation["status"],
        "hypothesis": HYPOTHESIS,
        "material_pass": evaluation["material_pass"],
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_balanced_accuracy_delta": evaluation["outer_balanced_accuracy_delta"],
        "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta_probability_gt_zero"],
        "candidate_metrics": evaluation["candidate_summary"]["metrics"],
        "selected_metrics": evaluation["selected_summary"]["metrics"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "material_checks": evaluation["material_checks"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "seal_authorized": False,
    }
    if args.smoke_test:
        print(json.dumps(clean({**summary, "status": "SMOKE_OK", "output_written": False}), indent=2))
        return
    json_reports, table_reports = build_reports(champion, evidence, audits, evaluation, input_audit)
    if args.dry_run:
        print(json.dumps(clean({**summary, "status": "DRY_RUN_OK", "output_written": False}), indent=2))
        return
    controller_result = write_outputs(json_reports, table_reports, evaluation, input_audit)
    print(json.dumps(clean({**summary, "output_dir": str(OUT), "controller": controller_result}), indent=2))


if __name__ == "__main__":
    main()
