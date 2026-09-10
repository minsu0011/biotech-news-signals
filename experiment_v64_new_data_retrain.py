"""V64: causal conservative adaptation to the role-frozen DEV extension.

Only two labelled inputs are authorized here:

* the immutable V36 DEV contract; and
* ``data/V59_DEV_EXTENSION_LABELED.csv.gz``, whose rows must all have the
  already-frozen ``DEV_EXTENSION`` role.

The research-seal and final-meta-reserve paths are intentionally absent from
this module.  Historical V58 outer-OOF predictions are preserved byte-for-row
in the combined result.  New rows receive strictly prequential predictions:
a source-specific surrogate distils the past V58 route, while a conservative
actual-label adapter may incorporate only earlier DEV-extension outcomes.
Blend weight and direction cutoff are chosen on an embargoed inner-past tail.
No current outer row participates in fitting, calibration, or confidence.
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
from scipy.special import expit, logit

import bio_news_30m_v3 as app
import experiment_v36_robust as v36
import experiment_v37_text as v37
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
OUT = Path(os.environ.get("MARKET_BIO_VERSION_OUTPUT", str(ROOT / "output_V64")))

V36_PATH = DATA / "dev_contract_v36_labeled.csv.gz"
V36_EVENTS_PATH = DATA / "dev_contract_v36_events.csv.gz"
EXTENSION_PATH = DATA / "V59_DEV_EXTENSION_LABELED.csv.gz"
V58_OOF_PATH = CACHE / "v58_locked_source_routing_oof.csv.gz"

# V64 is a reproducible experiment over the 38 rows available when its plan
# was declared.  A later-growing DEV-extension must become a new material
# version rather than silently changing this experiment.
EXPECTED_SHA256 = {
    V36_PATH.name: "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2",
    V36_EVENTS_PATH.name: "6b49d4545f4d2cc95f76c5bfbc27e4f2836b23b2e3ff2488473d0fd1a7d1e748",
    EXTENSION_PATH.name: "16267867e3a3898b10adb82b5d65b577d2d4a31f7d3a2c1f3f9bf69a336e13f7",
    V58_OOF_PATH.name: "8509a953f499ed4b9d9eebf33ff9ecb8909edb6bd41a8eb974f6d3963d4c665c",
}
EXPECTED_EXTENSION_ROWS = 38
ALLOWED_EXTENSION_ROLE = "DEV_EXTENSION"
EMBARGO = pd.Timedelta(minutes=35)
MODEL_PARAMETERS = dict(v36.FAMILIES["lightgbm"][0])
ADAPTER_WEIGHTS = (0.0, 0.10, 0.25, 0.40)
DIRECTION_CUTOFFS = (0.48, 0.50, 0.52)
RECENCY_HALF_LIFE_DAYS = 1460.0
EXTENSION_TRAIN_MULTIPLIER = 4.0
CONFIDENCE_QUANTILE = 0.85
MAX_PREQUENTIAL_BLOCKS = 4
EXTENSION_CLUSTER_POLICY = "STRICT_UNIQUE"
ALLOW_UNSCORABLE_PREFIX_QUARANTINE = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def verify_pinned_inputs() -> dict[str, str]:
    actual = {}
    for path in (V36_PATH, V36_EVENTS_PATH, EXTENSION_PATH, V58_OOF_PATH):
        require(path.is_file(), f"required V64 input missing: {path.name}")
        actual[path.name] = sha256(path)
        require(actual[path.name] == EXPECTED_SHA256[path.name],
                f"V64 pinned input changed: {path.name}")
    return actual


def exact_contract_audit(extension: pd.DataFrame) -> dict[str, Any]:
    required = {
        "event_id", "event_group_id", "market", "ticker", "event_time_utc",
        "source_family", "contract_version", "price_cadence_sec",
        "bar_timestamp_semantics", "entry_target_utc", "actual_entry_time_utc",
        "entry_slippage_sec", "exit_target_utc", "actual_exit_time_utc",
        "exit_slippage_sec", "actual_hold_seconds", "entry_price", "exit_price",
        "fwd_ret_30m", "y", "raw_file_sha256", "price_provider",
        "minutes_to_close", "data_role",
    }
    require(required.issubset(extension.columns),
            f"V64 extension schema missing: {sorted(required - set(extension.columns))}")
    require(len(extension) == EXPECTED_EXTENSION_ROWS,
            f"V64 extension row count changed: {len(extension)}")
    require(extension.event_id.is_unique, "V64 extension event_id is not unique")
    require(extension.event_group_id.is_unique, "V64 extension event_group_id is not unique")
    require(extension.data_role.eq(ALLOWED_EXTENSION_ROLE).all(),
            "V64 refuses any role other than DEV_EXTENSION")
    require(extension.contract_version.eq("EXACT_T2_30M_V36").all(),
            "V64 extension contract_version mismatch")
    require(extension.price_cadence_sec.le(60).all(), "V64 extension cadence is coarser than 1m")
    require(extension.bar_timestamp_semantics.eq("OPEN").all(),
            "V64 extension bar semantics mismatch")

    utc_columns = ("event_time_utc", "entry_target_utc", "actual_entry_time_utc",
                   "exit_target_utc", "actual_exit_time_utc")
    times = {column: pd.to_datetime(extension[column], utc=True, errors="raise")
             for column in utc_columns}
    entry_target_delta = (times["entry_target_utc"] - times["event_time_utc"]).dt.total_seconds()
    entry_slippage = (times["actual_entry_time_utc"] - times["entry_target_utc"]).dt.total_seconds()
    exit_target_delta = (times["exit_target_utc"] - times["actual_entry_time_utc"]).dt.total_seconds()
    exit_slippage = (times["actual_exit_time_utc"] - times["exit_target_utc"]).dt.total_seconds()
    actual_hold = (times["actual_exit_time_utc"] - times["actual_entry_time_utc"]).dt.total_seconds()
    require(entry_target_delta.eq(120).all(), "V64 extension entry target is not event+2m")
    require(entry_slippage.between(0, 60).all(), "V64 extension entry slippage violation")
    require(exit_target_delta.eq(1800).all(), "V64 extension exit target is not actual-entry+30m")
    require(exit_slippage.between(0, 60).all(), "V64 extension exit slippage violation")
    require(actual_hold.between(1800, 1860).all(), "V64 extension actual hold violation")
    require(np.allclose(entry_slippage, extension.entry_slippage_sec),
            "V64 extension recorded entry slippage mismatch")
    require(np.allclose(exit_slippage, extension.exit_slippage_sec),
            "V64 extension recorded exit slippage mismatch")
    require(np.allclose(actual_hold, extension.actual_hold_seconds),
            "V64 extension recorded hold mismatch")
    require(((extension.entry_price > 0) & (extension.exit_price > 0)).all(),
            "V64 extension contains non-positive prices")
    require(np.array_equal(extension.y.to_numpy(int),
                           (extension.fwd_ret_30m.to_numpy(float) > 0).astype(int)),
            "V64 extension y/return mismatch")
    require(extension.raw_file_sha256.fillna("").astype(str).str.fullmatch(r"[0-9a-f]{64}").all(),
            "V64 extension raw provenance SHA invalid")
    require(pd.to_numeric(extension.get("minutes_to_close"), errors="coerce").ge(31).all(),
            "V64 extension lacks a regular-session 30m exit window")

    return {
        "rows": int(len(extension)),
        "role": ALLOWED_EXTENSION_ROLE,
        "role_frozen": True,
        "markets": {str(k): int(v) for k, v in extension.market.value_counts().items()},
        "sources": {str(k): int(v) for k, v in extension.source_family.value_counts().items()},
        "providers": {str(k): int(v) for k, v in extension.price_provider.value_counts().items()},
        "event_time_min": times["event_time_utc"].min(),
        "event_time_max": times["event_time_utc"].max(),
        "entry_slippage_max_sec": float(entry_slippage.max()),
        "exit_slippage_max_sec": float(exit_slippage.max()),
        "actual_hold_min_sec": float(actual_hold.min()),
        "actual_hold_max_sec": float(actual_hold.max()),
        "contract_integrity": "PASS",
    }


def prepare_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pinned = verify_pinned_inputs()
    legacy = pd.read_csv(V36_PATH, compression="gzip", low_memory=False,
                         dtype={"ticker": str}, parse_dates=["event_time_utc"])
    events = pd.read_csv(V36_EVENTS_PATH, compression="gzip", low_memory=False,
                         usecols=["event_id", "url", "article_id"])
    legacy = legacy.merge(events, on="event_id", how="left", validate="one_to_one")
    extension = pd.read_csv(EXTENSION_PATH, compression="gzip", low_memory=False,
                            dtype={"ticker": str}, parse_dates=["event_time_utc"])
    audit = exact_contract_audit(extension)
    require(not extension.event_id.isin(set(legacy.event_id)).any(),
            "V64 extension overlaps V36 event_id")
    require(not extension.event_group_id.isin(set(legacy.event_group_id)).any(),
            "V64 extension overlaps V36 event_group_id")

    extension["url"] = ""
    extension["article_id"] = ""
    legacy["origin_role"] = "V36_DEV"
    extension["origin_role"] = ALLOWED_EXTENSION_ROLE
    for frame in (legacy, extension):
        frame["headline"] = frame.headline.fillna("").astype(str)
        frame["body"] = frame.body.fillna("").astype(str)
        frame["form"] = frame.form.fillna("").astype(str)
        frame["text_cluster_id"] = frame.apply(v37.normalized_cluster, axis=1)

    duplicate_cluster_rows = extension.text_cluster_id.duplicated(keep=False)
    if duplicate_cluster_rows.any():
        require(
            EXTENSION_CLUSTER_POLICY == "LABEL_BLIND_FIRST_EVENT_PER_CLUSTER",
            "V64 extension contains duplicate canonical text clusters",
        )
        before = len(extension)
        extension = (
            extension.sort_values(["event_time_utc", "event_id"], kind="stable")
            .drop_duplicates("text_cluster_id", keep="first")
            .copy()
        )
        audit["model_input_cluster_policy"] = EXTENSION_CLUSTER_POLICY
        audit["model_input_rows_after_cluster_quarantine"] = int(len(extension))
        audit["duplicate_cluster_rows_quarantined"] = int(before - len(extension))
        audit["cluster_representative_selection_uses_labels"] = False
    else:
        audit["model_input_cluster_policy"] = "STRICT_UNIQUE_PASS"
        audit["model_input_rows_after_cluster_quarantine"] = int(len(extension))
        audit["duplicate_cluster_rows_quarantined"] = 0
        audit["cluster_representative_selection_uses_labels"] = False
    require(not extension.text_cluster_id.isin(set(legacy.text_cluster_id)).any(),
            "V64 extension text cluster overlaps V36")
    combined = pd.concat([legacy, extension], ignore_index=True, sort=False)
    combined["cluster_weight"] = 1.0 / combined.groupby("text_cluster_id").event_id.transform("size")
    legacy = combined[combined.origin_role.eq("V36_DEV")].copy()
    extension = combined[combined.origin_role.eq(ALLOWED_EXTENSION_ROLE)].copy()

    v58 = pd.read_csv(V58_OOF_PATH, compression="gzip", low_memory=False,
                      dtype={"ticker": str}, parse_dates=["event_time_utc"])
    require(v58.event_id.is_unique, "V64 V58 teacher cache has duplicate event_id")
    require(v58.event_id.isin(set(legacy.event_id)).all(),
            "V64 V58 teacher cache contains non-V36 rows")
    required_v58 = {"prob", "high_conf", "confidence_signal", "fold", "family", "routed_from"}
    require(required_v58.issubset(v58.columns), "V64 V58 champion cache schema mismatch")
    require(len(v58) == 7568 and v58.family.eq("V58_LOCKED_SOURCE_DGP_ROUTING").all(),
            "V64 V58 champion row/family contract mismatch")
    expected_routes = {
        "US_SEC": "SEC_NUMERIC", "US_NEWS": "US_NEWS_MULTILINGUAL",
        "KR_NEWS": "KR_DIVERSE_CONSENSUS", "KR_KIND": "KR_DIVERSE_CONSENSUS",
    }
    for source, route in expected_routes.items():
        require(v58.loc[v58.source_family.eq(source), "routed_from"].eq(route).all(),
                f"V64 V58 route mismatch: {source}")
    v58_direction = v36.summary(v58)
    require(abs(v58_direction["balanced_accuracy"] - 0.525511873859576) < 1e-12 and
            abs(v58_direction["auc"] - 0.5229106843440796) < 1e-12,
            "V64 V58 champion metric contract mismatch")
    teacher = legacy.merge(v58[["event_id", "prob"]].rename(columns={"prob": "teacher_prob"}),
                           on="event_id", how="inner", validate="one_to_one")
    audit.update({
        "input_sha256": pinned,
        "v36_rows": int(len(legacy)),
        "v58_teacher_rows": int(len(teacher)),
        "v58_contract": {
            "status": "PASS", "rows": int(len(v58)), "fixed_routes": expected_routes,
            "balanced_accuracy": v58_direction["balanced_accuracy"],
            "auc": v58_direction["auc"],
            "confidence_policy": "frozen V48 outer-causal mask on historical V36 OOF",
        },
        "event_overlap_v36": 0,
        "event_group_overlap_v36": 0,
        "text_cluster_overlap_v36": 0,
        "authorized_label_inputs": [V36_PATH.name, EXTENSION_PATH.name],
        "forbidden_role_inputs_read": False,
    })
    return legacy, extension, v58, audit


def drift_audit(legacy: pd.DataFrame, extension: pd.DataFrame) -> dict[str, Any]:
    legacy_model = app.model_frame(legacy)
    extension_model = app.model_frame(extension)
    for column in ("market", "source_family", "ticker", "event_type", "form", "y",
                   "fwd_ret_30m"):
        legacy_model[column] = legacy[column].to_numpy()
        extension_model[column] = extension[column].to_numpy()
    rows = []
    for (market, source), current_index in extension.groupby(["market", "source_family"]).groups.items():
        current = extension_model.loc[current_index]
        prior = legacy_model[legacy_model.market.eq(market) & legacy_model.source_family.eq(source)]
        numeric = {}
        absolute = []
        for column in app.NUMERIC_COLS:
            old = pd.to_numeric(prior[column], errors="coerce")
            new = pd.to_numeric(current[column], errors="coerce")
            scale = float(old.std())
            smd = (float(new.mean()) - float(old.mean())) / scale if scale > 0 else float("nan")
            numeric[column] = finite_or_none(smd)
            if math.isfinite(smd):
                absolute.append(abs(smd))
        categorical = {}
        for column in ("ticker", "event_type", "form"):
            old_values = set(prior[column].fillna("").astype(str))
            values = current[column].fillna("").astype(str)
            categorical[f"unseen_{column}_rate"] = float((~values.isin(old_values)).mean())
        rows.append({
            "market": str(market), "source_family": str(source),
            "legacy_n": int(len(prior)), "extension_n": int(len(current)),
            "extension_prevalence": float(current.y.mean()),
            "extension_mean_return": float(current.fwd_ret_30m.mean()),
            "mean_abs_numeric_smd": float(np.mean(absolute)) if absolute else None,
            "max_abs_numeric_smd": float(np.max(absolute)) if absolute else None,
            "numeric_smd": numeric, **categorical,
        })
    return {
        "method": "standardized mean difference against same market/source V36 DEV",
        "by_source": rows,
        "extension_all_missing_model_features": [
            column for column in app.NUMERIC_COLS
            if extension_model[column].isna().all()
        ],
        "missing_feature_policy": (
            "label-independent derived features are recomputed by app.model_frame; only still-missing "
            "features use the past-training median inside the V36 pipeline; no future fill"
        ),
    }


def recency_weight(frame: pd.DataFrame, extension_multiplier: bool) -> np.ndarray:
    event_time = pd.to_datetime(frame.event_time_utc, utc=True)
    anchor = event_time.max()
    age_days = (anchor - event_time).dt.total_seconds().to_numpy(float) / 86400.0
    decay = np.maximum(0.10, np.power(0.5, age_days / RECENCY_HALF_LIFE_DAYS))
    weight = frame.cluster_weight.to_numpy(float) * decay
    if extension_multiplier:
        weight *= np.where(frame.origin_role.eq(ALLOWED_EXTENSION_ROLE),
                           EXTENSION_TRAIN_MULTIPLIER, 1.0)
    mean = float(np.mean(weight))
    return weight / mean if mean > 0 else np.ones(len(frame), float)


def make_direction_model():
    return v36.make_model("lightgbm", MODEL_PARAMETERS)


def fit_teacher(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    require(len(train) >= 200 and train.teacher_prob.notna().all(),
            "V64 teacher history insufficient")
    probability = np.clip(train.teacher_prob.to_numpy(float), 0.01, 0.99)
    base_weight = recency_weight(train.assign(origin_role="V36_DEV"), False)
    # Duplicated complementary labels optimize the binary cross-entropy for a
    # soft V58 target without inventing a regression preprocessing contract.
    expanded = pd.concat([train, train], ignore_index=True)
    labels = np.concatenate([np.ones(len(train), int), np.zeros(len(train), int)])
    weights = np.concatenate([base_weight * probability, base_weight * (1.0 - probability)])
    model = make_direction_model()
    model.fit(v36.feature_frame(expanded), labels, model__sample_weight=weights)
    return model.predict_proba(v36.feature_frame(target))[:, 1]


def fit_adapter(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    require(len(train) >= 200 and train.y.nunique() == 2,
            "V64 actual-label history insufficient")
    model = make_direction_model()
    model.fit(v36.feature_frame(train), train.y.astype(int),
              model__sample_weight=recency_weight(train, True))
    return model.predict_proba(v36.feature_frame(target))[:, 1]


def logit_blend(teacher: np.ndarray, adapter: np.ndarray, adapter_weight: float) -> np.ndarray:
    first = logit(np.clip(np.asarray(teacher, float), 1e-5, 1 - 1e-5))
    second = logit(np.clip(np.asarray(adapter, float), 1e-5, 1 - 1e-5))
    return expit((1.0 - adapter_weight) * first + adapter_weight * second)


def inner_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("event_time_utc").reset_index(drop=True)
    split = max(160, int(0.78 * len(ordered)))
    require(split < len(ordered) - 30, "V64 inner validation too small")
    valid = ordered.iloc[split:].copy()
    boundary = valid.event_time_utc.min()
    train = ordered[ordered.event_time_utc < boundary - EMBARGO].copy()
    require(len(train) >= 200 and train.y.nunique() == 2 and valid.y.nunique() == 2,
            "V64 inner chronological classes insufficient")
    return train, valid


def weighted_inner_metrics(frame: pd.DataFrame, probability: np.ndarray,
                           cutoff: float) -> dict[str, float]:
    weighted = frame.copy()
    role_multiplier = np.where(weighted.origin_role.eq(ALLOWED_EXTENSION_ROLE),
                               EXTENSION_TRAIN_MULTIPLIER, 1.0)
    weighted["cluster_weight"] = weighted.cluster_weight.to_numpy(float) * role_multiplier
    return v37.metric(weighted.reset_index(drop=True), probability, cutoff, weighted=True)


def confidence_values(reference_margin: np.ndarray, raw: np.ndarray, cutoff: float,
                      teacher: np.ndarray, adapter: np.ndarray) -> np.ndarray:
    margin_rank = v37.rank_against(reference_margin, np.abs(raw - cutoff))
    agreement = np.clip(1.0 - 2.0 * np.abs(teacher - adapter), 0.0, 1.0)
    return 0.75 * margin_rank + 0.25 * agreement


def choose_inner_policy(actual_history: pd.DataFrame, teacher_history: pd.DataFrame) -> dict[str, Any]:
    inner_train, inner_valid = inner_split(actual_history)
    boundary = inner_valid.event_time_utc.min()
    teacher_train = teacher_history[teacher_history.event_time_utc < boundary - EMBARGO].copy()
    teacher_probability = fit_teacher(teacher_train, inner_valid)
    adapter_probability = fit_adapter(inner_train, inner_valid)
    candidates = []
    for adapter_weight in ADAPTER_WEIGHTS:
        raw = logit_blend(teacher_probability, adapter_probability, adapter_weight)
        for cutoff in DIRECTION_CUTOFFS:
            metrics = weighted_inner_metrics(inner_valid, raw, float(cutoff))
            checks = int(metrics["balanced_accuracy"] >= 0.51) + int(metrics["auc"] >= 0.51) + \
                int(metrics["edge"] >= 0.0)
            score = (2.0 * metrics["balanced_accuracy"] + 1.5 * metrics["auc"] +
                     metrics["edge"])
            margin = np.abs(raw - cutoff)
            margin_rank = v37.rank_self(margin)
            agreement = np.clip(1.0 - 2.0 * np.abs(teacher_probability - adapter_probability), 0, 1)
            confidence = 0.75 * margin_rank + 0.25 * agreement
            candidates.append({
                "adapter_weight": float(adapter_weight), "direction_cutoff": float(cutoff),
                "checks": checks, "score": float(score), "inner_metrics": metrics,
                "confidence_threshold": float(np.quantile(confidence, CONFIDENCE_QUANTILE)),
                "margin_reference": margin,
            })
    # A tie deliberately favours less adaptation and a cutoff nearer 0.5.
    candidates.sort(key=lambda row: (row["checks"], row["score"],
                                     -row["adapter_weight"],
                                     -abs(row["direction_cutoff"] - 0.5)), reverse=True)
    best = candidates[0]
    return {
        **{key: value for key, value in best.items() if key != "margin_reference"},
        "margin_reference": best["margin_reference"],
        "inner_train_n": int(len(inner_train)), "inner_valid_n": int(len(inner_valid)),
        "inner_extension_n": int(inner_valid.origin_role.eq(ALLOWED_EXTENSION_ROLE).sum()),
        "inner_train_end": inner_train.event_time_utc.max(),
        "inner_valid_start": inner_valid.event_time_utc.min(),
    }


def extension_blocks(frame: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = frame.sort_values(["event_time_utc", "event_id"]).reset_index(drop=True)
    count = min(MAX_PREQUENTIAL_BLOCKS, len(ordered))
    return [ordered.iloc[index].copy() for index in np.array_split(np.arange(len(ordered)), count)
            if len(index)]


def prequential_extension(legacy: pd.DataFrame, extension: pd.DataFrame,
                          v58: pd.DataFrame, smoke: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    teacher_all = legacy.merge(
        v58[["event_id", "prob"]].rename(columns={"prob": "teacher_prob"}),
        on="event_id", how="inner", validate="one_to_one")
    outputs = []
    audits = []
    quarantined_ids: list[str] = []
    groups = list(extension.groupby(["market", "source_family"], sort=True))
    for (market, source), source_extension in groups:
        blocks = extension_blocks(source_extension)
        if smoke:
            blocks = blocks[:1]
        legacy_source = legacy[legacy.market.eq(market) & legacy.source_family.eq(source)].copy()
        teacher_source = teacher_all[
            teacher_all.market.eq(market) & teacher_all.source_family.eq(source)].copy()
        for subfold, valid in enumerate(blocks, 1):
            boundary = valid.event_time_utc.min()
            legacy_past = legacy_source[legacy_source.event_time_utc < boundary - EMBARGO].copy()
            extension_past = source_extension[
                source_extension.event_time_utc < boundary - EMBARGO].copy()
            actual_history = pd.concat([legacy_past, extension_past], ignore_index=True, sort=False)
            teacher_history = teacher_source[
                teacher_source.event_time_utc < boundary - EMBARGO].copy()
            if actual_history.empty or teacher_history.empty:
                require(
                    ALLOW_UNSCORABLE_PREFIX_QUARANTINE,
                    "V64 outer chronology/embargo violation",
                )
                ids = sorted(valid.event_id.astype(str))
                quarantined_ids.extend(ids)
                audits.append({
                    "market": str(market), "source_family": str(source),
                    "subfold": int(subfold), "valid_n": int(len(valid)),
                    "status": "QUARANTINED_NO_CAUSAL_PAST_TRAINING_HISTORY",
                    "valid_start": boundary, "embargo_minutes": 35,
                    "selection_uses_labels": False,
                    "event_id_set_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                    "current_outer_labels_used": False,
                })
                continue
            require(not valid.event_id.isin(set(actual_history.event_id)).any(),
                    "V64 current outer outcome entered training")
            require(actual_history.event_time_utc.max() < boundary - EMBARGO,
                    "V64 outer chronology/embargo violation")
            policy = choose_inner_policy(actual_history, teacher_history)
            teacher_probability = fit_teacher(teacher_history, valid)
            adapter_probability = fit_adapter(actual_history, valid)
            raw = logit_blend(teacher_probability, adapter_probability,
                              float(policy["adapter_weight"]))
            confidence = confidence_values(policy["margin_reference"], raw,
                                           float(policy["direction_cutoff"]),
                                           teacher_probability, adapter_probability)
            high = confidence >= float(policy["confidence_threshold"])
            probability = app.shift_probability(raw, float(policy["direction_cutoff"]))
            seen = set(actual_history.ticker.astype(str))
            output = valid[["event_id", "event_group_id", "event_time_utc", "market", "ticker",
                            "source_family", "form", "y", "fwd_ret_30m", "text_cluster_id",
                            "cluster_weight"]].copy()
            output["prob"] = probability
            output["high_conf"] = high
            output["confidence_signal"] = confidence
            output["fold"] = 5
            output["family"] = "V64_V58_TEACHER_NEW_DATA_ADAPTER"
            output["new_issuer"] = ~output.ticker.astype(str).isin(seen)
            output["prob_champion_teacher"] = teacher_probability
            output["prob_actual_adapter"] = adapter_probability
            output["adapt_weight"] = float(policy["adapter_weight"])
            output["direction_cutoff"] = float(policy["direction_cutoff"])
            output["extension_subfold"] = int(subfold)
            output["routed_from"] = "V64_CHAMPION_DISTILLATION_ADAPTER"
            outputs.append(output)
            audit_policy = {key: value for key, value in policy.items()
                            if key != "margin_reference"}
            audits.append({
                "market": str(market), "source_family": str(source),
                "subfold": int(subfold), "valid_n": int(len(valid)),
                "legacy_past_n": int(len(legacy_past)),
                "extension_past_n": int(len(extension_past)),
                "teacher_past_n": int(len(teacher_history)),
                "train_end": actual_history.event_time_utc.max(),
                "valid_start": boundary, "embargo_minutes": 35,
                "policy": audit_policy, "current_outer_labels_used": False,
            })
            print(f"[V64 ADAPT] {market}/{source} subfold={subfold} valid={len(valid)} "
                  f"prior_extension={len(extension_past)} weight={policy['adapter_weight']:.2f} "
                  f"cutoff={policy['direction_cutoff']:.2f}", flush=True)
    require(outputs, "V64 produced no extension predictions")
    result = pd.concat(outputs, ignore_index=True)
    require(result.event_id.is_unique, "V64 duplicate extension predictions")
    if not smoke and not quarantined_ids:
        require(set(result.event_id) == set(extension.event_id),
                "V64 did not predict every authorized extension row")
    if quarantined_ids:
        require(
            set(result.event_id) | set(quarantined_ids) == set(extension.event_id),
            "V64 predicted/quarantined event partition mismatch",
        )
        audits.append({
            "status": "LABEL_BLIND_PREFIX_QUARANTINE_SUMMARY",
            "quarantined_rows": len(quarantined_ids),
            "event_id_set_sha256": hashlib.sha256(
                "\n".join(sorted(quarantined_ids)).encode()
            ).hexdigest(),
            "selection_uses_labels": False,
        })
    require(np.isfinite(result.prob).all() and result.prob.between(0, 1).all(),
            "V64 invalid extension probabilities")
    return result, audits


def extension_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = v36.summary(frame)
    by_source = {}
    for (market, source), group in frame.groupby(["market", "source_family"]):
        by_source[f"{market}|{source}"] = v36.summary(group, include_subgroups=False)
    return {
        "metrics": metrics,
        "by_market_source_even_when_small": by_source,
        "official_gate_eligible": False,
        "reason": "role-frozen DEV extension has only 38 rows; diagnostic only, never a seal",
    }


def extension_component(frame: pd.DataFrame, probability_column: str,
                        family: str) -> pd.DataFrame:
    component = frame.copy()
    probability = np.clip(component[probability_column].to_numpy(float), 1e-6, 1 - 1e-6)
    cutoff = np.clip(component.direction_cutoff.to_numpy(float), 1e-6, 1 - 1e-6)
    component["prob"] = expit(logit(probability) - logit(cutoff))
    component["family"] = family
    return component


def add_missing_v58_columns(extension: pd.DataFrame, v58: pd.DataFrame) -> pd.DataFrame:
    result = extension.copy()
    bool_columns = {column for column in v58.columns if column.startswith("high_")}
    for column in v58.columns:
        if column in result.columns:
            continue
        result[column] = False if column in bool_columns else np.nan
    return result[list(dict.fromkeys([*v58.columns, *[c for c in result.columns if c not in v58.columns]]))]


def execute(smoke: bool = False) -> dict[str, Any]:
    legacy, extension, v58, source_audit = prepare_frames()
    source_audit["drift"] = drift_audit(legacy, extension)
    extension_oof, fold_audits = prequential_extension(legacy, extension, v58, smoke=smoke)
    if smoke:
        return {
            "mode": "SMOKE_TEST", "status": "PASS", "predicted_rows": int(len(extension_oof)),
            "sources": sorted(extension_oof.source_family.unique().tolist()),
            "fold_audits": fold_audits,
            "source_audit": source_audit,
        }

    historical = v58.copy()
    extension_for_combined = add_missing_v58_columns(extension_oof, historical)
    combined = pd.concat([historical, extension_for_combined], ignore_index=True, sort=False)
    require(combined.event_id.is_unique, "V64 combined OOF duplicate event")
    result = v44.summarize(combined)
    historical_result = v44.summarize(historical)
    extension_result = extension_diagnostics(extension_oof)
    teacher_extension_result = extension_diagnostics(extension_component(
        extension_oof, "prob_champion_teacher", "V64_EXTENSION_TEACHER_ONLY"))
    adapter_extension_result = extension_diagnostics(extension_component(
        extension_oof, "prob_actual_adapter", "V64_EXTENSION_ADAPTER_ONLY"))
    status = "ROBUST_SURVIVOR" if result["research_gate"]["robust_survivor"] else "RESEARCH_FAIL"
    validation = (
        "V58 historical outer-OOF rows frozen unchanged; V64 extension uses source-specific V58 soft-target "
        "distillation plus an actual-label recency adapter; every extension block is predicted from rows ending "
        ">=35 minutes earlier; only prior DEV_EXTENSION outcomes may enter the adapter; blend/cutoff are selected "
        "on an embargoed inner-past tail; RESEARCH_SEAL_POOL and FINAL_META_RESERVE are never read"
    )
    comparison = {
        "version": "V64", "hypothesis": "NEW_DATA_RETRAIN_V1",
        "material_change": "role-frozen new exact data enters a causal prequential adapter",
        "input_sha256": source_audit["input_sha256"],
        "configuration": {
            "model": "V36 LightGBM candidate 0",
            "teacher": "source-specific V58 soft-target distillation",
            "adapter_weights": ADAPTER_WEIGHTS, "direction_cutoffs": DIRECTION_CUTOFFS,
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "extension_train_multiplier": EXTENSION_TRAIN_MULTIPLIER,
            "confidence_quantile": CONFIDENCE_QUANTILE,
        },
        "models": {
            "V64_COMBINED_OOF": result,
            "V58_FROZEN_HISTORY": historical_result,
            "V64_EXTENSION_PREQUENTIAL": extension_result,
            "V64_EXTENSION_TEACHER_ONLY": teacher_extension_result,
            "V64_EXTENSION_ADAPTER_ONLY": adapter_extension_result,
        },
        "fold_audits": fold_audits, "source_audit": source_audit,
        "validation": validation,
    }
    robustness = {
        "version": "V64", "status": status, "hypothesis": "NEW_DATA_RETRAIN_V1",
        "opened_dev_only": True, "v36plus_seal_outcomes_loaded": False,
        "selected": result, "extension_diagnostic": extension_result,
        "new_data_count": int(len(extension_oof)),
        "new_data_by_market": {str(k): int(v) for k, v in extension_oof.market.value_counts().items()},
        "fold_audits": fold_audits, "validation": validation, "seal_authorized": False,
    }
    transfer = {
        "version": "V64", "seal_outcomes_loaded": False,
        "selected_oof_by_source": result["metrics"]["by_source_family"],
        "selected_oof_by_market": result["metrics"]["by_market"],
        "selected_oof_new_issuer": result["metrics"]["new_issuer"],
        "extension_diagnostic": extension_result,
        "source_drift": source_audit["drift"], "method": validation,
    }
    return {
        "mode": "FULL", "status": status, "combined": combined,
        "extension_oof": extension_oof, "comparison": comparison,
        "robustness": robustness, "transfer": transfer, "source_audit": source_audit,
    }


def write_result(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    v37.atomic_json(payload["comparison"], OUT / "MODEL_COMPARISON.json")
    v37.atomic_json(payload["robustness"], OUT / "DEV_ROBUSTNESS_REPORT.json")
    v37.atomic_json(payload["transfer"], OUT / "SOURCE_TRANSFER_REPORT.json")
    v37.atomic_json(payload["source_audit"], OUT / "DEV_EXTENSION_AUDIT.json")
    payload["combined"].to_csv(CACHE / "v64_new_data_retrain_oof.csv.gz", index=False,
                               compression="gzip")
    payload["extension_oof"].to_csv(CACHE / "v64_extension_prequential_oof.csv.gz", index=False,
                                    compression="gzip")


def public_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["mode"] == "SMOKE_TEST":
        return {key: value for key, value in payload.items() if key != "source_audit"}
    selected = payload["robustness"]["selected"]
    extension = payload["robustness"]["extension_diagnostic"]
    return {
        "version": "V64", "status": payload["status"],
        "combined_metrics": selected["metrics"],
        "research_gate": selected["research_gate"],
        "extension_metrics": extension["metrics"],
        "extension_official_gate_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V64 causal new-data retrain")
    parser.add_argument("--audit-only", action="store_true",
                        help="verify only the pinned authorized inputs; write nothing")
    parser.add_argument("--smoke-test", action="store_true",
                        help="fit the first prequential block per source; write nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every prequential block and metric; write nothing")
    args = parser.parse_args()
    runtime_limits.configure()
    if args.audit_only:
        legacy, extension, _, audit = prepare_frames()
        audit["drift"] = drift_audit(legacy, extension)
        print(json.dumps({"version": "V64", "mode": "AUDIT_ONLY", "status": "PASS",
                          "audit": audit}, ensure_ascii=False, indent=2, default=str))
        return
    payload = execute(smoke=args.smoke_test)
    if not (args.smoke_test or args.dry_run):
        write_result(payload)
    print(json.dumps(public_summary(payload), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
