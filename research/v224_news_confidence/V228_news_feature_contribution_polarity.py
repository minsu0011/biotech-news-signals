"""V228 strict nested contribution-polarity audit for P_CORRECT_NEWS.

One fixed correctness-model architecture is evaluated source-specifically on
old immutable DEV.  In each nested chronological fold, paired full and
block-ablated models share the same train/validation samples and transformed
features.  Inner folds alone select a contribution dosage from the fixed grid
[-1, 0, .5, 1, 1.5]; outer labels are diagnostic only.

No Research Seal, Final Meta, DEV_EXTENSION, role assignment, registry, or
state is opened or modified.  Results are not independent confirmation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import V227_news_direction_sample_weighting as v227


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "V228_POLARITY_AUDIT"
ALPHAS = (-1.0, 0.0, 0.5, 1.0, 1.5)
ALPHA_TIE_PREFERENCE = {1.0: 5, 0.5: 4, 0.0: 3, -1.0: 2, 1.5: 1}
BLOCKS = (
    "PUBLISHER", "TOPIC", "SOURCE_CONSENSUS", "SOURCE_DIVERSITY", "NOVELTY",
    "STRUCTURED_CONFIRMATION", "MODEL_AGREEMENT", "MARKET_CONTEXT", "TEXT_CERTAINTY",
)
NEWS_SOURCES = ("US_NEWS", "KR_NEWS")
HC_QUANTILE = 0.80
STABILITY_REQUIRED = 0.80
COST = 0.002
SEED = 22801
BOOTSTRAP_DRAWS = 500
MATERIAL_BUDGET = 1
EXPECTED_V227_RUNNER_SHA256 = "98d606bb2002ebaaba918581f659d855af8e039875095f91d18192029f476cc6"

PRIOR_MANIFEST_SHA256 = {
    "V225_STRICT_OOF": "4be273610dad4700bcb4c793c55bd6bd8c76c6ccc6b90ca2854bc7e246796788",
    "V226_US_DUAL_GATE": "19be79ccaa0d5455f70439840d4041ffdae4fb7d224bee6107b515eaf2b0fa1b",
    "V227_DIRECTION_WEIGHTING": "7001687e9ffd3de13d95dd219989dc6880d2af4a5e3bb89dbe88500db15b3d74",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            compressed.write(raw)
    temporary.replace(path)


def verify_manifest(directory: Path, expected_hash: str) -> dict[str, Any]:
    manifest_path = directory / "MANIFEST.json"
    require(sha256(manifest_path) == expected_hash,
            f"V228_PRIOR_MANIFEST_CHANGED:{directory.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    for name, spec in files.items():
        path = directory / name
        require(path.is_file(), f"V228_PRIOR_FILE_MISSING:{directory.name}:{name}")
        require(path.stat().st_size == int(spec["bytes"]),
                f"V228_PRIOR_BYTES_CHANGED:{directory.name}:{name}")
        require(sha256(path) == spec["sha256"],
                f"V228_PRIOR_HASH_CHANGED:{directory.name}:{name}")
    return {
        "manifest_sha256": expected_hash, "files_verified": len(files),
        "path": str(directory.relative_to(ROOT)),
    }


def verify_authority() -> dict[str, Any]:
    base = v227.verify_authority()
    prior = {
        name: verify_manifest(HERE / name, digest)
        for name, digest in PRIOR_MANIFEST_SHA256.items()
    }
    v227_runner_hash = sha256(HERE / "V227_news_direction_sample_weighting.py")
    require(v227_runner_hash == EXPECTED_V227_RUNNER_SHA256,
            "V228_V227_RUNNER_CHANGED")
    return {
        "base": base, "prior_evidence": prior,
        "v227_runner_sha256": v227_runner_hash,
        "old_dev_only": True, "research_seal_loaded": False,
        "final_meta_loaded": False, "state_loaded_or_modified": False,
    }


def numeric(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        name: pd.to_numeric(frame[name], errors="coerce") for name in names
    }, index=frame.index)


def text_certainty(frame: pd.DataFrame) -> pd.DataFrame:
    headline = frame.headline.fillna("").astype(str)
    body = frame.body.fillna("").astype(str)
    combined = headline + " " + body
    length = combined.str.len().clip(lower=1).astype(float)
    return pd.DataFrame({
        "headline_chars": headline.str.len().astype(float),
        "headline_words": headline.str.count(r"\S+").astype(float),
        "body_chars": body.str.len().astype(float),
        "body_words": body.str.count(r"\S+").astype(float),
        "digit_fraction": combined.str.count(r"[0-9]").astype(float) / length,
        "uppercase_fraction": combined.str.count(r"[A-Z]").astype(float) / length,
        "question_count": combined.str.count(r"\?").astype(float),
        "exclamation_count": combined.str.count("!").astype(float),
        "positive_negative_clash": (
            frame.positive_semantic.astype(bool) & frame.negative_semantic.astype(bool)
        ).astype(float),
        "semantic_certainty": np.abs(
            frame.positive_semantic.to_numpy(float) - frame.negative_semantic.to_numpy(float)
        ),
    }, index=frame.index)


def raw_blocks(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    v225 = v227.get_v225()
    publisher = pd.DataFrame(index=frame.index)
    for name in v225.PUBLISHER_CLASSES:
        publisher[f"publisher__{name}"] = frame.publisher_class.eq(name).astype(float)
    publisher["timestamp_exact"] = (
        frame.timestamp_quality.fillna("").astype(str).str.upper().eq("EXACT").astype(float)
    )
    topic_columns = [name for name in frame.columns if name.startswith("topic__")]
    topic = numeric(frame, topic_columns)
    topic["positive_semantic"] = frame.positive_semantic.to_numpy(float)
    topic["negative_semantic"] = frame.negative_semantic.to_numpy(float)
    agreement = pd.DataFrame(index=frame.index)
    direction = frame.prob.to_numpy(float) >= 0.5
    agreement["direction_margin"] = np.abs(frame.prob.to_numpy(float) - 0.5)
    agreement["predicted_up"] = direction.astype(float)
    for name in v225.EXPERT_CONFIDENCE:
        agreement[f"confidence__{name}"] = pd.to_numeric(frame[name], errors="coerce")
    for name in v225.EXPERT_PROBS:
        probability = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        agreement[f"margin__{name}"] = np.abs(probability - 0.5)
        agreement[f"agrees__{name}"] = ((probability >= 0.5) == direction).astype(float)
    blocks = {
        "PUBLISHER": publisher,
        "TOPIC": topic,
        "SOURCE_CONSENSUS": numeric(frame, (
            "decision_news_rows", "decision_mean_similarity", "decision_min_similarity",
            "decision_topic_agreement", "decision_contradiction",
        )),
        "SOURCE_DIVERSITY": numeric(frame, (
            "decision_source_clusters", "decision_publisher_entropy",
            "decision_source_type_diversity",
        )),
        "NOVELTY": numeric(frame, (
            "novelty_count_24h", "novelty_count_7d", "novelty_count_30d",
            "novelty_max_similarity_7d",
        )),
        "STRUCTURED_CONFIRMATION": numeric(frame, ("structured_confirmation",)),
        "MODEL_AGREEMENT": agreement,
        "MARKET_CONTEXT": numeric(frame, v225.MARKET_COLUMNS),
        "TEXT_CERTAINTY": text_certainty(frame),
    }
    require(tuple(blocks) == BLOCKS, "V228_BLOCK_REGISTRY_DRIFT")
    forbidden = v227.FORBIDDEN_DIRECTION_FEATURES
    require(not any(set(value.columns) & forbidden for value in blocks.values()),
            "V228_FORBIDDEN_FEATURE")
    return blocks


def transformed_blocks(
    train: pd.DataFrame, target: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    train_raw = raw_blocks(train)
    target_raw = raw_blocks(target)
    train_output: dict[str, Any] = {}
    target_output: dict[str, Any] = {}
    counts = {}
    for block in BLOCKS:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        scaler = StandardScaler()
        train_numeric = scaler.fit_transform(imputer.fit_transform(train_raw[block]))
        target_numeric = scaler.transform(imputer.transform(target_raw[block]))
        train_matrix: Any = sparse.csr_matrix(train_numeric)
        target_matrix: Any = sparse.csr_matrix(target_numeric)
        if block == "TEXT_CERTAINTY":
            vectorizer = TfidfVectorizer(
                lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=3,
                max_df=0.995, max_features=24000, sublinear_tf=True, dtype=np.float32,
            )
            train_text = vectorizer.fit_transform(train.news_text)
            target_text = vectorizer.transform(target.news_text)
            train_matrix = sparse.hstack([train_matrix, train_text], format="csr")
            target_matrix = sparse.hstack([target_matrix, target_text], format="csr")
        train_output[block] = train_matrix
        target_output[block] = target_matrix
        counts[block] = int(train_matrix.shape[1])
    return train_output, target_output, counts


def stack(block_matrices: dict[str, Any], excluded: str | None = None) -> Any:
    names = [name for name in BLOCKS if name != excluded]
    return sparse.hstack([block_matrices[name] for name in names], format="csr")


def fit_logit_score(matrix: Any, target_matrix: Any, frame: pd.DataFrame, seed: int) -> np.ndarray:
    model = LogisticRegression(
        C=0.25, solver="liblinear", class_weight="balanced", max_iter=1000,
        random_state=seed,
    )
    model.fit(
        matrix, frame.correct_target.to_numpy(int),
        sample_weight=v227.local_cluster_weight(frame),
    )
    return model.decision_function(target_matrix).astype(float)


def paired_contributions(
    train: pd.DataFrame, target: pd.DataFrame, seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    train_blocks, target_blocks, counts = transformed_blocks(train, target)
    full_train = stack(train_blocks)
    full_target = stack(target_blocks)
    z_full = fit_logit_score(full_train, full_target, train, seed)
    output = {}
    for index, block in enumerate(BLOCKS):
        z_ablated = fit_logit_score(
            stack(train_blocks, excluded=block), stack(target_blocks, excluded=block),
            train, seed + index + 1,
        )
        output[block] = (z_full.copy(), z_ablated)
    return output, {
        "train_n": len(train), "target_n": len(target),
        "feature_counts": counts, "full_feature_n": int(full_train.shape[1]),
        "paired_same_train_target": True,
        "model": "L2_LOGISTIC_C_0.25_BALANCED_CLUSTER_WEIGHTED",
        "target": "immutable V69 outer-OOF direction correctness",
    }


def alpha_probability(z_full: np.ndarray, z_ablated: np.ndarray, alpha: float) -> np.ndarray:
    delta = z_full - z_ablated
    return np.clip(expit(z_ablated + alpha * delta), 1e-6, 1.0 - 1e-6)


def safe_auc(target: Iterable[int], probability: Iterable[float]) -> float | None:
    y = np.asarray(list(target), dtype=int)
    score = np.asarray(list(probability), dtype=float)
    return float(roc_auc_score(y, score)) if len(y) and np.unique(y).size == 2 else None


def safe_ba(target: Iterable[int], prediction: Iterable[bool]) -> float | None:
    y = np.asarray(list(target), dtype=int)
    value = np.asarray(list(prediction), dtype=bool)
    return float(balanced_accuracy_score(y, value)) if len(y) and np.unique(y).size == 2 else None


def inner_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    count = max(1, int(math.ceil(0.20 * len(target))))
    chosen = np.argsort(-probability, kind="stable")[:count]
    return {
        "auc": safe_auc(target, probability),
        "balanced_accuracy": safe_ba(target, probability >= 0.5),
        "top20_accuracy": float(target[chosen].mean()),
    }


def choose_alpha(rows: list[dict[str, Any]]) -> float:
    def key(row: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            -math.inf if row["inner_auc"] is None else float(row["inner_auc"]),
            -math.inf if row["inner_balanced_accuracy"] is None
            else float(row["inner_balanced_accuracy"]),
            float(row["inner_top20_accuracy"]), ALPHA_TIE_PREFERENCE[float(row["alpha"])],
        )
    return float(max(rows, key=key)["alpha"])


def alpha_rows(
    valid: pd.DataFrame, source: str, outer_fold: int, block: str,
    z_full: np.ndarray, z_ablated: np.ndarray, thresholds: dict[float, float],
    chosen_alpha: float,
) -> pd.DataFrame:
    delta = z_full - z_ablated
    pieces = []
    for alpha in ALPHAS:
        probability = alpha_probability(z_full, z_ablated, alpha)
        high = probability >= thresholds[alpha]
        output = valid[[
            "event_id", "event_time_utc", "ticker", "source_family", "fold",
            "y", "fwd_ret_30m", "correct_target", "signed_net", "text_cluster_id",
        ]].copy()
        output["outer_fold"] = outer_fold
        output["feature_block"] = block
        output["alpha"] = alpha
        output["chosen_by_inner"] = alpha == chosen_alpha
        output["p_correct_news"] = probability
        output["high_conf"] = high
        output["hc_threshold"] = thresholds[alpha]
        output["z_full"] = z_full
        output["z_ablated"] = z_ablated
        output["delta_logit"] = delta
        output["abs_delta_logit"] = np.abs(delta)
        output["abs_return"] = valid.fwd_ret_30m.abs().to_numpy(float)
        output["outer_labels_used_for_alpha_or_threshold"] = False
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True)


def run_source(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    source = str(frame.source_family.iloc[0])
    evidence = []
    selection_rows = []
    pair_audits = []
    for train, valid, outer_fold, outer_audit in v227.outer_folds(frame):
        inner_train, inner_valid, inner_audit = v227.inner_split(train)
        inner_pairs, inner_model_audit = paired_contributions(
            inner_train, inner_valid, SEED + outer_fold * 1000
        )
        outer_pairs, outer_model_audit = paired_contributions(
            train, valid, SEED + outer_fold * 1000 + 100
        )
        for block in BLOCKS:
            inner_full, inner_ablated = inner_pairs[block]
            candidate_rows = []
            thresholds = {}
            for alpha in ALPHAS:
                probability = alpha_probability(inner_full, inner_ablated, alpha)
                values = inner_metrics(inner_valid.correct_target.to_numpy(int), probability)
                thresholds[alpha] = float(np.quantile(probability, HC_QUANTILE))
                candidate_rows.append({
                    "source_family": source, "outer_fold": outer_fold,
                    "feature_block": block, "alpha": alpha,
                    "inner_auc": values["auc"],
                    "inner_balanced_accuracy": values["balanced_accuracy"],
                    "inner_top20_accuracy": values["top20_accuracy"],
                    "inner_hc_threshold": thresholds[alpha],
                    "outer_labels_used": False,
                })
            chosen = choose_alpha(candidate_rows)
            for row in candidate_rows:
                row["chosen_alpha"] = chosen
                row["chosen_by_inner"] = float(row["alpha"]) == chosen
            selection_rows.extend(candidate_rows)
            outer_full, outer_ablated = outer_pairs[block]
            evidence.append(alpha_rows(
                valid, source, outer_fold, block, outer_full, outer_ablated,
                thresholds, chosen,
            ))
            pair_audits.append({
                "source_family": source, "outer_fold": outer_fold,
                "feature_block": block, **outer_audit, **inner_audit,
                "inner_full_feature_n": inner_model_audit["full_feature_n"],
                "outer_full_feature_n": outer_model_audit["full_feature_n"],
                "block_feature_n_inner": inner_model_audit["feature_counts"][block],
                "block_feature_n_outer": outer_model_audit["feature_counts"][block],
                "paired_same_train_target": True,
                "same_row_fitted_correctness": False,
                "outer_labels_used_for_alpha_or_threshold": False,
                "alpha_grid": "-1,0,0.5,1,1.5",
            })
        print(
            f"[V228] {source} fold={outer_fold}/5 train={len(train)} valid={len(valid)}",
            flush=True,
        )
    return (
        pd.concat(evidence, ignore_index=True), pd.DataFrame(selection_rows), pair_audits
    )


def bootstrap_net(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    work = frame.copy().reset_index(drop=True)
    work["day"] = pd.to_datetime(work.event_time_utc, utc=True).dt.date
    groups = {day: np.asarray(index, int) for day, index in work.groupby("day").indices.items()}
    days = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        positions = np.concatenate([groups[day] for day in rng.choice(days, len(days), replace=True)])
        sample = work.iloc[positions]
        selected = sample.high_conf.to_numpy(bool)
        if selected.any():
            values.append(float(sample.loc[selected, "signed_net"].mean()))
    require(values, "V228_BOOTSTRAP_EMPTY")
    return {
        "draws": len(values), "lower95": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)), "upper95": float(np.quantile(values, 0.975)),
        "probability_gt_zero": float(np.mean(np.asarray(values) > 0.0)),
        "method": "event-day block bootstrap of frozen nested-OOF policy",
    }


def score_metrics(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    target = frame.correct_target.to_numpy(int)
    probability = frame.p_correct_news.to_numpy(float)
    high = frame.high_conf.to_numpy(bool)
    hc = frame.loc[high]
    require(len(hc) > 5, "V228_HC_SUPPORT")
    net = np.sort(hc.signed_net.to_numpy(float))
    daily = hc.assign(day=pd.to_datetime(hc.event_time_utc, utc=True).dt.date).groupby("day")
    daily_net = daily.signed_net.mean()
    return {
        "n": len(frame), "confidence_correctness_auc": safe_auc(target, probability),
        "balanced_accuracy": safe_ba(target, probability >= 0.5),
        "hc_n": int(high.sum()), "hc_coverage": float(high.mean()),
        "hc_accuracy": float(hc.correct_target.mean()),
        "hc_net": float(hc.signed_net.mean()),
        "hc_abs_return": float(hc.abs_return.mean()),
        "hc_winsorized_1_99_net": float(np.clip(net, *np.quantile(net, [0.01, 0.99])).mean()),
        "hc_top1_positive_removed_net": float(net[:-1].mean()),
        "hc_top5_positive_removed_net": float(net[:-5].mean()),
        "bootstrap_hc_net": bootstrap_net(frame, seed),
        "hc_date_n": int(len(daily_net)),
        "hc_positive_date_fraction": float((daily_net > 0.0).mean()),
        "mean_abs_delta_logit": float(frame.abs_delta_logit.mean()),
        "p95_abs_delta_logit": float(frame.abs_delta_logit.quantile(0.95)),
    }


def confidence_deciles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = frame.sort_values("p_correct_news", kind="mergesort").reset_index(drop=True)
    ordered["decile"] = np.minimum(9, np.arange(len(ordered)) * 10 // len(ordered)) + 1
    return [{
        "decile": int(decile), "n": len(part),
        "mean_p_correct": float(part.p_correct_news.mean()),
        "correctness": float(part.correct_target.mean()),
        "mean_net": float(part.signed_net.mean()),
        "mean_abs_return": float(part.abs_return.mean()),
        "mean_abs_delta_logit": float(part.abs_delta_logit.mean()),
    } for decile, part in ordered.groupby("decile", sort=True)]


def contribution_extremes(frame: pd.DataFrame) -> list[dict[str, Any]]:
    labels = ("0_50", "50_80", "80_90", "90_95", "95_100")
    fractions = np.asarray([0.50, 0.80, 0.90, 0.95, 1.00])
    ordered = frame.sort_values("abs_delta_logit", kind="mergesort").reset_index(drop=True)
    percentile = (np.arange(len(ordered)) + 1) / len(ordered)
    ordered["magnitude_bin"] = [labels[int(np.searchsorted(fractions, value, side="left"))]
                                for value in percentile]
    return [{
        "magnitude_bin": label, "n": len(part),
        "mean_abs_delta_logit": float(part.abs_delta_logit.mean()),
        "mean_p_correct": float(part.p_correct_news.mean()),
        "correctness": float(part.correct_target.mean()),
        "correctness_auc": safe_auc(part.correct_target, part.p_correct_news),
        "hc_coverage": float(part.high_conf.mean()),
        "hc_accuracy": float(part.loc[part.high_conf, "correct_target"].mean())
            if part.high_conf.any() else None,
        "mean_net": float(part.signed_net.mean()),
        "mean_abs_return": float(part.abs_return.mean()),
    } for label, part in ordered.groupby("magnitude_bin", sort=False)]


def classification(selection: pd.DataFrame) -> dict[str, Any]:
    chosen = selection[selection.chosen_by_inner.astype(bool)].copy()
    counts = chosen.chosen_alpha.value_counts().to_dict()
    max_count = max(counts.values())
    modes = [float(alpha) for alpha, count in counts.items() if count == max_count]
    if len(modes) > 1:
        means = selection.groupby("alpha").inner_auc.mean().to_dict()
        modal = max(modes, key=lambda alpha: (means[alpha], ALPHA_TIE_PREFERENCE[alpha]))
    else:
        modal = modes[0]
    stability = max_count / len(chosen)
    if stability < STABILITY_REQUIRED:
        label = "UNSTABLE"
    elif modal == -1.0:
        label = "POLARITY_REVERSED"
    elif modal == 0.0:
        label = "REDUNDANT_OR_HARMFUL"
    elif modal == 0.5:
        label = "OVERWEIGHTED_OR_SATURATING"
    elif modal == 1.0:
        label = "SUPPORTIVE"
    else:
        label = "AMPLIFICATION_FAVORED"
    return {
        "classification": label, "modal_inner_alpha": modal,
        "stability": stability, "support_outer_folds": len(chosen),
        "inner_chosen_counts": {str(float(key)): int(value) for key, value in counts.items()},
        "selection_basis": "inner folds only",
    }


def build_outputs(
    evidence: pd.DataFrame, selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_rows = []
    decile_rows = []
    extreme_rows = []
    selected_pieces = []
    classifications: dict[str, Any] = {}
    seed_offset = 0
    for source in NEWS_SOURCES:
        classifications[source] = {}
        for block in BLOCKS:
            group_selection = selection[
                selection.source_family.eq(source) & selection.feature_block.eq(block)
            ]
            class_result = classification(group_selection)
            classifications[source][block] = class_result
            block_evidence = evidence[
                evidence.source_family.eq(source) & evidence.feature_block.eq(block)
            ]
            selected = block_evidence[block_evidence.chosen_by_inner.astype(bool)].copy()
            require(selected.event_id.is_unique, "V228_SELECTED_EVENT_DUPLICATE")
            selected_pieces.append(selected)
            selected_metrics = score_metrics(selected, SEED + seed_offset)
            score_rows.append({
                "source_family": source, "feature_block": block,
                "policy": "INNER_SELECTED", "alpha": class_result["modal_inner_alpha"],
                **class_result, **selected_metrics,
            })
            seed_offset += 1
            for alpha in ALPHAS:
                part = block_evidence[block_evidence.alpha.eq(alpha)].copy()
                metrics = score_metrics(part, SEED + 1000 + seed_offset)
                score_rows.append({
                    "source_family": source, "feature_block": block,
                    "policy": "FIXED_ALPHA_OUTER_DIAGNOSTIC", "alpha": alpha,
                    **class_result, **metrics,
                })
                seed_offset += 1
                for row in confidence_deciles(part):
                    decile_rows.append({
                        "source_family": source, "feature_block": block,
                        "alpha": alpha, **row,
                    })
                extremes = contribution_extremes(part)
                bottom = next(row for row in extremes if row["magnitude_bin"] == "0_50")
                top = next(row for row in extremes if row["magnitude_bin"] == "95_100")
                reversal = (
                    top["mean_p_correct"] > bottom["mean_p_correct"]
                    and top["correctness"] < bottom["correctness"]
                )
                for row in extremes:
                    extreme_rows.append({
                        "source_family": source, "feature_block": block,
                        "alpha": alpha, "extreme_contribution_reversal": reversal, **row,
                    })
    return (
        pd.DataFrame(score_rows), pd.DataFrame(decile_rows), pd.DataFrame(extreme_rows),
        pd.concat(selected_pieces, ignore_index=True), classifications,
    )


def ledger_records(classifications: dict[str, Any]) -> list[dict[str, Any]]:
    records = [{
        "record_id": "V228_OLD_DEV_PROMOTION_READY",
        "hypothesis": "old-DEV contribution polarity is sufficient for promotion",
        "decision": "REJECTED_BY_CONTRACT",
        "reason": "independent confirmation and untouched evaluation are absent",
        "namespace": "V228", "material_budget": MATERIAL_BUDGET,
    }]
    for source in NEWS_SOURCES:
        for block in BLOCKS:
            result = classifications[source][block]
            if result["classification"] == "UNSTABLE":
                records.append({
                    "record_id": f"V228_{source}_{block}_STABLE_POLARITY",
                    "hypothesis": f"{source}/{block} has stable contribution polarity",
                    "decision": "REJECTED_INNER_INSTABILITY",
                    "reason": f"modal agreement {result['stability']:.3f} < {STABILITY_REQUIRED:.2f}",
                    "namespace": "V228", "material_budget": MATERIAL_BUDGET,
                })
            elif result["classification"] == "SUPPORTIVE":
                records.append({
                    "record_id": f"V228_{source}_{block}_REVERSAL_OR_SHRINK",
                    "hypothesis": f"{source}/{block} requires reversal or shrinkage",
                    "decision": "REJECTED_INNER_SELECTION",
                    "reason": "stable inner modal alpha is +1.0",
                    "namespace": "V228", "material_budget": MATERIAL_BUDGET,
                })
    return records


def append_ledger_once(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["record_id"]] = row
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            identifier = record["record_id"]
            if identifier in existing:
                require(existing[identifier] == record, f"V228_LEDGER_CONFLICT:{identifier}")
                continue
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_report(
    authority: dict[str, Any], runtime: dict[str, Any], evidence: pd.DataFrame,
    selection: pd.DataFrame, pair_audits: list[dict[str, Any]],
    scorecard: pd.DataFrame, classifications: dict[str, Any],
) -> dict[str, Any]:
    require(not evidence.outer_labels_used_for_alpha_or_threshold.astype(bool).any(),
            "V228_OUTER_SELECTION_LEAKAGE")
    require(not selection.outer_labels_used.astype(bool).any(), "V228_INNER_AUDIT_FLAG")
    require(all(row["text_cluster_overlap_n"] == 0 for row in pair_audits),
            "V228_OUTER_CLUSTER_OVERLAP")
    require(all(row["inner_text_cluster_overlap_n"] == 0 for row in pair_audits),
            "V228_INNER_CLUSTER_OVERLAP")
    selected = scorecard[scorecard.policy.eq("INNER_SELECTED")]
    return {
        "version": 228,
        "status": "OLD_DEV_NESTED_DIAGNOSTIC_ONLY_NO_PROMOTION",
        "hypothesis": "P_CORRECT_NEWS_FEATURE_CONTRIBUTION_POLARITY",
        "authority": authority, "runtime": runtime,
        "pre_registered_budget": {
            "material_experiments": MATERIAL_BUDGET,
            "blocks": list(BLOCKS), "alpha_grid": list(ALPHAS),
            "model": "L2_LOGISTIC_C_0.25_BALANCED_CLUSTER_WEIGHTED",
            "alpha_selection": "inner AUC, then BA, then top20 correctness, fixed tie preference",
            "hc_gate": "inner q80 P_CORRECT score for each alpha",
            "stability_required": STABILITY_REQUIRED,
            "micro_feature_tuning": False, "micro_threshold_search": False,
        },
        "contract": {
            "source_specific": list(NEWS_SOURCES), "outer_folds": 5,
            "chronological_embargo_minutes": 35.0, "text_cluster_purge": True,
            "paired_full_ablated_same_fold": True,
            "delta_definition": "z_full - z_without_block",
            "alpha_score": "z_without_block + alpha * delta",
            "outer_labels_used_for_alpha_or_threshold": False,
            "same_row_fitted_correctness": False,
            "direction": "immutable V69 outer-OOF; confidence role only",
        },
        "inventory": {
            "alpha_outer_rows": len(evidence),
            "unique_outer_events": int(evidence.event_id.nunique()),
            "inner_selection_rows": len(selection),
            "paired_fold_audit_rows": len(pair_audits),
        },
        "classifications": classifications,
        "inner_selected_summary": selected.to_dict("records"),
        "audit": {
            "outer_label_selection_reference_n": 0,
            "inner_text_cluster_overlap_n": 0, "outer_text_cluster_overlap_n": 0,
            "same_row_correctness_reference_n": 0,
            "classification_uses_inner_only": True,
        },
        "limitations": [
            "OLD_DEV_ONLY_NOT_INDEPENDENT_CONFIRMATION",
            "NO_CAPTURE_OR_INGEST_TIMESTAMP_FOR_TEXT",
            "OUTER_RESULTS_ARE_DIAGNOSTIC_NOT_SELECTION",
            "NO_PRODUCTION_PROMOTION_AUTHORIZED",
        ],
        "research_seal_or_final_rows_read": False, "production_authorized": False,
    }


def audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V228 Nested P_CORRECT_NEWS Contribution-Polarity Audit", "",
        f"Status: `{report['status']}`", "",
        "One fixed material experiment evaluated nine pre-registered feature blocks. "
        "Alpha and HC thresholds were frozen from inner chronological folds only.", "",
        "## Inner-only polarity classifications", "",
    ]
    for source in NEWS_SOURCES:
        for block in BLOCKS:
            row = report["classifications"][source][block]
            lines.append(
                f"- {source}/{block}: {row['classification']}, modal alpha="
                f"{row['modal_inner_alpha']}, stability={row['stability']:.3f}."
            )
    lines.extend([
        "", "Outer confidence deciles, contribution extremes, date-block bootstrap, "
        "winsorized net, and positive-tail removal are evaluation diagnostics only.", "",
        "No independent confirmation, Research Seal, or Final Meta claim is made.", "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-ids", default="30,31")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cpu_ids = tuple(int(piece.strip()) for piece in args.cpu_ids.split(",") if piece.strip())
    runtime = v227.configure_runtime(cpu_ids)
    authority = verify_authority()
    news = v227.load_news()
    runtime = v227.configure_runtime(cpu_ids)
    all_evidence = []
    all_selection = []
    all_pair_audits = []
    for source in NEWS_SOURCES:
        evidence, selection, audits = run_source(news[news.source_family.eq(source)].copy())
        all_evidence.append(evidence)
        all_selection.append(selection)
        all_pair_audits.extend(audits)
    evidence = pd.concat(all_evidence, ignore_index=True)
    selection = pd.concat(all_selection, ignore_index=True)
    scorecard, deciles, extremes, selected_oof, classifications = build_outputs(
        evidence, selection
    )
    report = build_report(
        authority, runtime, evidence, selection, all_pair_audits, scorecard, classifications
    )
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv_gz(evidence, OUT / "V228_ALPHA_OUTER_OOF.csv.gz")
    atomic_csv_gz(selected_oof, OUT / "V228_INNER_SELECTED_OOF.csv.gz")
    atomic_csv_gz(selection, OUT / "V228_INNER_SELECTION_AUDIT.csv.gz")
    atomic_csv_gz(pd.DataFrame(all_pair_audits), OUT / "V228_PAIRED_FOLD_AUDIT.csv.gz")
    atomic_csv_gz(scorecard, OUT / "V228_POLARITY_SCORECARD.csv.gz")
    atomic_csv_gz(deciles, OUT / "V228_CONFIDENCE_DECILES.csv.gz")
    atomic_csv_gz(extremes, OUT / "V228_CONTRIBUTION_EXTREMES.csv.gz")
    atomic_json(report, OUT / "V228_POLARITY_REPORT.json")
    markdown = OUT / "V228_POLARITY_AUDIT.md"
    temporary = markdown.with_name(markdown.name + f".tmp.{os.getpid()}")
    temporary.write_text(audit_markdown(report), encoding="utf-8")
    temporary.replace(markdown)
    ledger_path = OUT / "V228_REJECTED_HYPOTHESES_LEDGER.jsonl"
    append_ledger_once(ledger_records(classifications), ledger_path)
    files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "MANIFEST.json"
    }
    test_path = HERE / "V228_test_news_feature_contribution_polarity.py"
    manifest = {
        "version": 228, "status": report["status"],
        "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
        "tests": {"path": str(test_path.relative_to(ROOT)), "sha256": sha256(test_path)},
        "files": files, "prior_evidence": authority["prior_evidence"],
        "invariants": {
            "outer_labels_used_for_alpha_or_threshold": False,
            "paired_full_ablated_same_fold": True,
            "same_row_fitted_correctness": False,
            "existing_v225_v226_v227_modified": False,
            "research_seal_or_final_rows_read": False,
            "independent_confirmation_claimed": False,
        },
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps(json_clean({
        "status": report["status"], "classifications": classifications,
        "inner_selected_summary": report["inner_selected_summary"],
        "files": files, "manifest_sha256": sha256(OUT / "MANIFEST.json"),
    }), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
