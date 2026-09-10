"""V80 temporal heterogeneous graph direction challenger.

Each embargoed training cut creates a new typed graph whose nodes are source
family, event taxonomy, filing form, market, and causal pre-event regime.
Training events form label-independent co-occurrence edges.  Shrunk signed
direction state is initialized at nodes and then propagated through relation-
normalized one- and two-hop messages.  There is no ticker node, text feature,
semantic retrieval, embedding, or reusable graph fitted beyond the cut.

The graph interactions, rather than a direct categorical target-prior model,
are the material information: only propagated 1/2-hop states are candidates.
An inner-past OOF slice chooses exact V69 or one of three fixed graph blends.
Outer outcomes are evaluation-only.  Authority is immutable V36 DEV plus the
atomic output_V69 commit.  Material failure returns the exact complete V69
frame and the controller recomputes all 14 canonical gates.
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
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import autonomous_v37plus as controller
import experiment_v44_microstructure as v44
import runtime_limits


ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "data" / "dev_contract_v36_labeled.csv.gz"
V69_DIR = ROOT / "output_V69"
V69_PATH = V69_DIR / "V69_FAIL_CLOSED_SELECTED_OOF.csv.gz"
CONTROLLER_OUTPUT = os.environ.get("MARKET_BIO_VERSION_OUTPUT")
DEFAULT_OUT = (
    Path(CONTROLLER_OUTPUT)
    if CONTROLLER_OUTPUT
    else ROOT / "staging" / "V80_TEMPORAL_HETEROGENEOUS_GRAPH_V1"
)

VERSION = 80
HYPOTHESIS = "TEMPORAL_HETEROGENEOUS_EVENT_GRAPH_PROPAGATION_V1"
EXPECTED_V36_SHA256 = "2152090f9c83f233f0abe0fcb4fdb4b1a50cee27da99b27865f8eda83c8d65e2"
EXPECTED_V69_EXPERIMENT_ID = "445e21fc752036c96446e571d0c182a99624d1ab261ea9db72e14911d3d30740"
EXPECTED_V36_ROWS = 9462
EXPECTED_V69_ROWS = 7568
EXPECTED_MARKET_FOLD_ROWS = {"US": 1595, "KR": 297}
EXPECTED_NODE_TYPES = ("SOURCE_FAMILY", "EVENT_TAXONOMY", "FORM", "MARKET", "REGIME")
EMBARGO = pd.Timedelta(minutes=35)
INNER_VALID_FRACTION = 0.30
NODE_SHRINK = 18.0
MESSAGE_ANCHOR = 1.5
SEED = 8001
BOOTSTRAP_DRAWS = 2000
COST = 0.002

ARCHITECTURES = (
    {"name": "GRAPH_ONE_HOP_W0.35", "graph": "one_hop", "challenger_weight": 0.35},
    {"name": "GRAPH_TWO_HOP_W0.35", "graph": "two_hop", "challenger_weight": 0.35},
    {"name": "GRAPH_ONE_TWO_CONSENSUS_W0.50", "graph": "consensus", "challenger_weight": 0.50},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def number_or(value: Any, default: float) -> float:
    result = finite(value)
    return float(default) if result is None else result


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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.map(lambda item: str(item).strip().lower() in {"1", "true", "yes"}).astype(bool)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, int)
    return float(roc_auc_score(target, np.asarray(score, float))) if np.unique(target).size == 2 else None


def atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_bytes((json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"), path)


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


def verify_authority() -> dict[str, Any]:
    commit_path = V69_DIR / "COMMIT.json"
    manifest_path = V69_DIR / "ARTIFACT_MANIFEST.json"
    require(sha256(DEV_PATH) == EXPECTED_V36_SHA256, "immutable V36 DEV hash changed")
    require(commit_path.is_file() and manifest_path.is_file(), "V69 atomic authority incomplete")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(commit.get("version") == 69, "V69 COMMIT version mismatch")
    require(commit.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 experiment changed")
    require(commit.get("manifest_sha256") == sha256(manifest_path), "V69 manifest binding changed")
    require(commit.get("data_sha") == EXPECTED_V36_SHA256, "V69 not bound to immutable V36 DEV")
    require(manifest.get("experiment_id") == EXPECTED_V69_EXPERIMENT_ID, "V69 manifest experiment changed")
    for name, item in manifest.get("files", {}).items():
        path = V69_DIR / name
        require(path.is_file(), f"V69 manifested artifact missing: {name}")
        require(path.stat().st_size == int(item["bytes"]), f"V69 artifact size changed: {name}")
        require(sha256(path) == item["sha256"], f"V69 artifact hash changed: {name}")
    require(V69_PATH.name in manifest.get("files", {}), "V69 selected OOF not manifested")
    return {
        "authority": "immutable V36 DEV + atomic output_V69",
        "authorized_inputs": [
            str(DEV_PATH.relative_to(ROOT)), str(commit_path.relative_to(ROOT)),
            str(manifest_path.relative_to(ROOT)), str(V69_PATH.relative_to(ROOT)),
        ],
        "v36_sha256": EXPECTED_V36_SHA256,
        "v69_commit_sha256": sha256(commit_path),
        "v69_manifest_sha256": sha256(manifest_path),
        "v69_oof_sha256": sha256(V69_PATH),
        "manifest_files_verified": len(manifest.get("files", {})),
        "failed_output_loaded": False,
        "dev_extension_loaded": False,
        "role_assignment_loaded": False,
        "research_seal_loaded": False,
        "final_reserve_loaded": False,
    }


def load_authorized() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(
        DEV_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    champion = pd.read_csv(
        V69_PATH, compression="gzip", low_memory=False, dtype={"ticker": str},
        parse_dates=["event_time_utc"],
    )
    require(len(dev) == EXPECTED_V36_ROWS and len(champion) == EXPECTED_V69_ROWS, "authority row count changed")
    require(dev.event_id.is_unique and champion.event_id.is_unique, "event_id uniqueness failed")
    require(set(champion.event_id).issubset(set(dev.event_id)), "V69 is not a V36 subset")
    aligned = dev.set_index("event_id").loc[champion.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion.y.to_numpy(int)), "V36/V69 labels differ")
    require(np.allclose(aligned.fwd_ret_30m, champion.fwd_ret_30m, rtol=0.0, atol=1e-15), "V36/V69 returns differ")
    dev["event_time_utc"] = pd.to_datetime(dev.event_time_utc, utc=True)
    champion["event_time_utc"] = pd.to_datetime(champion.event_time_utc, utc=True)
    champion["high_conf"] = bool_series(champion.high_conf)
    for column in ("event_type", "form", "source_family", "market"):
        dev[column] = dev[column].fillna("UNKNOWN").astype(str)
    return dev.sort_values(["event_time_utc", "event_id"], kind="stable").reset_index(drop=True), champion


def event_taxonomy(row: pd.Series) -> str:
    event = str(row.event_type or "OTHER").upper().strip()
    if event.startswith("SEC_"):
        return "SEC_REPORTING"
    flags = []
    for column, name in (
        ("event_regulatory_kw", "REGULATORY"), ("event_trial_kw", "CLINICAL"),
        ("event_financing_kw", "FINANCING"), ("event_ma_kw", "MA"),
        ("event_earnings_kw", "EARNINGS"), ("event_negative_kw", "NEGATIVE"),
        ("event_positive_kw", "POSITIVE"),
    ):
        value = finite(row.get(column))
        if value is not None and value > 0:
            flags.append(name)
    if event not in {"", "UNKNOWN", "OTHER"}:
        return event
    return "+".join(flags[:3]) if flags else "OTHER"


def category_node(kind: str, value: Any) -> str:
    return f"{kind}::{str(value or 'UNKNOWN').upper().strip()}"


class TemporalHeterogeneousGraph:
    def __init__(self):
        self.regime_edges: dict[str, tuple[float, float]] = {}
        self.global_signed = 0.0
        self.node_base: dict[str, float] = {}
        self.node_support: dict[str, float] = {}
        self.adjacency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.hop_one: dict[str, float] = {}
        self.hop_two: dict[str, float] = {}
        self.audit: dict[str, Any] = {}

    @staticmethod
    def node_type(node: str) -> str:
        return node.split("::", 1)[0]

    def add_regime(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        output = frame.copy()
        definitions = {
            "VOL": pd.to_numeric(output.pre_vol_30, errors="coerce"),
            "ACT": pd.to_numeric(output.volume_ratio_5_30, errors="coerce"),
            "TREND": pd.to_numeric(output.trend_slope_30, errors="coerce"),
        }
        labels = []
        for name, values in definitions.items():
            if fit:
                valid = values[np.isfinite(values)]
                require(len(valid) >= 200, f"{name} regime support insufficient")
                self.regime_edges[name] = tuple(float(v) for v in np.quantile(valid, [1 / 3, 2 / 3]))
            low, high = self.regime_edges[name]
            labels.append(np.where(~np.isfinite(values), "X", np.where(values <= low, "L", np.where(values <= high, "M", "H"))))
        minute = pd.to_numeric(output.minutes_from_open, errors="coerce").fillna(-1).to_numpy(float)
        session = np.where(minute < 0, "X", np.where(minute < 90, "OPEN", np.where(minute < 270, "MID", "CLOSE")))
        output["graph_regime"] = [f"VOL_{v}|ACT_{a}|TREND_{t}|SESSION_{s}" for v, a, t, s in zip(*labels, session)]
        output["graph_event_taxonomy"] = output.apply(event_taxonomy, axis=1)
        output["graph_form"] = output.form.replace({"": "NEWS", "UNKNOWN": "NEWS"}).fillna("NEWS").astype(str)
        return output

    @staticmethod
    def nodes(row: pd.Series) -> tuple[str, ...]:
        return (
            category_node("SOURCE_FAMILY", row.source_family),
            category_node("EVENT_TAXONOMY", row.graph_event_taxonomy),
            category_node("FORM", row.graph_form),
            category_node("MARKET", row.market),
            category_node("REGIME", row.graph_regime),
        )

    def propagate(self, previous: dict[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for node, neighbors in self.adjacency.items():
            by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for neighbor, edge_weight in neighbors.items():
                by_type[self.node_type(neighbor)].append((edge_weight, previous.get(neighbor, self.global_signed)))
            type_messages = []
            for values in by_type.values():
                denominator = sum(weight for weight, _ in values)
                if denominator > 0:
                    type_messages.append(sum(weight * state for weight, state in values) / denominator)
            neighbor_message = float(np.mean(type_messages)) if type_messages else self.global_signed
            output[node] = float((MESSAGE_ANCHOR * self.node_base[node] + neighbor_message) / (MESSAGE_ANCHOR + 1.0))
        return output

    def fit(self, train: pd.DataFrame) -> "TemporalHeterogeneousGraph":
        prepared = self.add_regime(train, True)
        group_count = prepared.groupby("event_group_id").event_id.transform("size").to_numpy(float)
        weights = 1.0 / np.maximum(group_count, 1.0)
        signed = 2.0 * prepared.y.to_numpy(float) - 1.0
        self.global_signed = float(np.average(signed, weights=weights))
        signed_sum: dict[str, float] = defaultdict(float)
        support: dict[str, float] = defaultdict(float)
        typed_counts: dict[str, int] = defaultdict(int)
        for position, (_, row) in enumerate(prepared.iterrows()):
            nodes = self.nodes(row)
            weight = float(weights[position])
            for node in nodes:
                signed_sum[node] += weight * signed[position]
                support[node] += weight
            for left_index, left in enumerate(nodes):
                for right in nodes[left_index + 1:]:
                    self.adjacency[left][right] += weight
                    self.adjacency[right][left] += weight
        for node, count in support.items():
            self.node_base[node] = float((signed_sum[node] + NODE_SHRINK * self.global_signed) / (count + NODE_SHRINK))
            self.node_support[node] = float(count)
            typed_counts[self.node_type(node)] += 1
        require(tuple(sorted(typed_counts)) == tuple(sorted(EXPECTED_NODE_TYPES)), "graph node types changed")
        self.hop_one = self.propagate(self.node_base)
        self.hop_two = self.propagate(self.hop_one)
        edge_count = int(sum(len(neighbors) for neighbors in self.adjacency.values()) // 2)
        hop_one_changed = int(sum(not np.isclose(self.hop_one[node], self.node_base[node]) for node in self.node_base))
        hop_two_changed = int(sum(not np.isclose(self.hop_two[node], self.hop_one[node]) for node in self.node_base))
        require(edge_count > len(self.node_base) and hop_one_changed > 0 and hop_two_changed > 0, "graph propagation is degenerate")
        self.audit = {
            "train_n": len(train), "node_count": len(self.node_base), "edge_count": edge_count,
            "nodes_by_type": dict(sorted(typed_counts.items())),
            "directed_relation_types": sorted({f"{self.node_type(a)}->{self.node_type(b)}" for a, neighbors in self.adjacency.items() for b in neighbors}),
            "hop_one_changed_nodes": hop_one_changed, "hop_two_changed_nodes": hop_two_changed,
            "global_signed_direction": self.global_signed,
            "node_shrink": NODE_SHRINK, "message_anchor": MESSAGE_ANCHOR,
            "ticker_node": False, "text_node": False, "semantic_retrieval": False,
            "graph_fitted_on_training_cut_only": True,
        }
        return self

    def predict(self, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        prepared = self.add_regime(target, False)
        one_scores = []
        two_scores = []
        unseen = 0
        total = 0
        for _, row in prepared.iterrows():
            nodes = self.nodes(row)
            total += len(nodes)
            unseen += sum(node not in self.node_base for node in nodes)
            one_scores.append(float(np.mean([self.hop_one.get(node, self.global_signed) for node in nodes])))
            two_scores.append(float(np.mean([self.hop_two.get(node, self.global_signed) for node in nodes])))
        one_probability = np.clip((np.asarray(one_scores) + 1.0) / 2.0, 1e-5, 1.0 - 1e-5)
        two_probability = np.clip((np.asarray(two_scores) + 1.0) / 2.0, 1e-5, 1.0 - 1e-5)
        return one_probability, two_probability, {
            **self.audit, "target_n": len(target),
            "target_node_instances": total, "unseen_target_node_instances": unseen,
            "unseen_target_node_fraction": float(unseen / max(1, total)),
            "one_hop_probability_sha256": array_sha256(one_probability),
            "two_hop_probability_sha256": array_sha256(two_probability),
        }


def probability_blend(baseline: np.ndarray, challenger: np.ndarray, weight: float) -> np.ndarray:
    base_logit = logit(np.clip(np.asarray(baseline, float), 1e-5, 1.0 - 1e-5))
    graph_logit = logit(np.clip(np.asarray(challenger, float), 1e-5, 1.0 - 1e-5))
    return expit((1.0 - weight) * base_logit + weight * graph_logit)


def architecture_probability(baseline: np.ndarray, one: np.ndarray, two: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    if spec["graph"] == "one_hop":
        graph = one
    elif spec["graph"] == "two_hop":
        graph = two
    else:
        graph = expit(0.5 * logit(np.clip(one, 1e-5, 1.0 - 1e-5)) + 0.5 * logit(np.clip(two, 1e-5, 1.0 - 1e-5)))
    return probability_blend(baseline, graph, float(spec["challenger_weight"]))


def metric(frame: pd.DataFrame, probability: np.ndarray, confidence: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, float)
    prediction = probability >= 0.5
    y = frame.y.to_numpy(int)
    confidence = np.asarray(confidence, float)
    high = np.asarray(high, bool)
    correct = (prediction == y).astype(int)
    signed_net = np.where(prediction, 1.0, -1.0) * frame.fwd_ret_30m.to_numpy(float) - COST
    return {
        "n": len(frame), "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(correct.mean()), "pred_up": float(prediction.mean()),
        "all_trade_mean_signed_net": float(signed_net.mean()),
        "confidence_correctness_auc": safe_auc(correct, confidence),
        "highconf_n": int(high.sum()),
        "highconf_accuracy": float(correct[high].mean()) if high.any() else None,
        "highconf_mean_signed_net": float(signed_net[high].mean()) if high.any() else None,
    }


def chronology(train: pd.DataFrame, valid: pd.DataFrame, label: str) -> dict[str, Any]:
    require(len(train) and len(valid), f"{label}: empty split")
    train_end = pd.Timestamp(train.event_time_utc.max())
    valid_start = pd.Timestamp(valid.event_time_utc.min())
    require(train_end < valid_start - EMBARGO, f"{label}: strict 35-minute embargo failed")
    overlap = set(train.event_group_id.astype(str)) & set(valid.event_group_id.astype(str))
    require(not overlap, f"{label}: event-group leakage")
    return {
        "train_n": len(train), "valid_n": len(valid),
        "train_end": train_end, "valid_start": valid_start,
        "embargo_minutes": EMBARGO.total_seconds() / 60.0,
        "event_group_overlap_n": 0, "strict_35m_embargo": True,
    }


def prior_train(dev: pd.DataFrame, boundary: pd.Timestamp, valid: pd.DataFrame) -> pd.DataFrame:
    train = dev.loc[dev.event_time_utc < boundary - EMBARGO].copy()
    train = train.loc[~train.event_group_id.astype(str).isin(set(valid.event_group_id.astype(str)))].copy()
    return train


def aligned_dev(dev: pd.DataFrame, champion_rows: pd.DataFrame) -> pd.DataFrame:
    aligned = dev.set_index("event_id").loc[champion_rows.event_id].reset_index()
    require(np.array_equal(aligned.y.to_numpy(int), champion_rows.y.to_numpy(int)), "aligned labels differ")
    return aligned


def inner_partition(dev: pd.DataFrame, champion: pd.DataFrame, market: str, outer_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior = champion.loc[champion.market.eq(market) & (champion.event_time_utc < outer_start - EMBARGO)].copy()
    prior = prior.sort_values(["event_time_utc", "event_id"], kind="stable")
    minimum = 220 if market == "KR" else 1000
    require(len(prior) >= minimum, f"insufficient prior {market} OOF support")
    split = int(math.floor(len(prior) * (1.0 - INNER_VALID_FRACTION)))
    inner_champion = prior.iloc[split:].copy()
    inner_valid = aligned_dev(dev, inner_champion)
    inner_start = pd.Timestamp(inner_valid.event_time_utc.min())
    inner_train = prior_train(dev, inner_start, inner_valid)
    require(len(inner_train) >= 500, "inner graph training cut too small")
    audit = chronology(inner_train, inner_valid, f"V80 inner {market}")
    audit["validation_source"] = "earlier same-market committed V69 OOF IDs"
    audit["selection_uses_inner_labels_only"] = True
    return inner_train, inner_valid, inner_champion, audit


def choose_inner(inner_train: pd.DataFrame, inner_valid: pd.DataFrame, inner_champion: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = TemporalHeterogeneousGraph().fit(inner_train)
    one, two, graph_audit = graph.predict(inner_valid)
    baseline = inner_champion.prob.to_numpy(float)
    confidence = inner_champion.confidence_signal.to_numpy(float)
    high = bool_series(inner_champion.high_conf).to_numpy(bool)
    baseline_metric = metric(inner_valid, baseline, confidence, high)
    trials = [{
        "name": "V69_NOOP", "spec": None, "metrics": baseline_metric,
        "auc_delta": 0.0, "balanced_accuracy_delta": 0.0,
        "all_trade_net_delta": 0.0, "eligible": True,
        "score": float(2.0 * baseline_metric["auc"] + baseline_metric["balanced_accuracy"] + 10.0 * baseline_metric["all_trade_mean_signed_net"]),
    }]
    for spec in ARCHITECTURES:
        probability = architecture_probability(baseline, one, two, spec)
        values = metric(inner_valid, probability, confidence, high)
        auc_delta = values["auc"] - baseline_metric["auc"]
        ba_delta = values["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        net_delta = values["all_trade_mean_signed_net"] - baseline_metric["all_trade_mean_signed_net"]
        trials.append({
            "name": spec["name"], "spec": spec, "metrics": values,
            "auc_delta": auc_delta, "balanced_accuracy_delta": ba_delta,
            "all_trade_net_delta": net_delta,
            "eligible": bool(ba_delta >= -0.005 and net_delta >= -0.0005),
            "score": float(2.0 * values["auc"] + values["balanced_accuracy"] + 10.0 * values["all_trade_mean_signed_net"]),
        })
    selected = max(
        (trial for trial in trials if trial["eligible"]),
        key=lambda trial: (trial["score"], trial["auc_delta"], trial["all_trade_net_delta"], trial["name"] == "V69_NOOP"),
    )
    return {
        "selection_rule": "inner-past OOF only; maximize 2*AUC+BA+10*all-trade-net with BA delta>=-0.005 and net delta>=-0.0005; exact no-op and three fixed propagated graph blends",
        "baseline": baseline_metric, "selected": selected, "trials": trials,
        "outer_labels_used_for_selection": False,
    }, graph_audit


def run_nested(dev: pd.DataFrame, champion: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    diagnostic = champion.copy()
    diagnostic["v80_model"] = "V69_NOOP"
    evidence_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    fold_specs = [(market, fold) for market in ("US", "KR") for fold in (2, 3, 4)]
    for market, fold in fold_specs[:1] if smoke else fold_specs:
        outer_champion = champion.loc[champion.market.eq(market) & champion.fold.eq(fold)].copy()
        outer_champion = outer_champion.sort_values(["event_time_utc", "event_id"], kind="stable")
        require(len(outer_champion) == EXPECTED_MARKET_FOLD_ROWS[market], f"{market} fold {fold} row count changed")
        outer_valid = aligned_dev(dev, outer_champion)
        outer_start = pd.Timestamp(outer_valid.event_time_utc.min())
        outer_train = prior_train(dev, outer_start, outer_valid)
        outer_chronology = chronology(outer_train, outer_valid, f"V80 outer {market} fold {fold}")
        inner_train, inner_valid, inner_champion, inner_chronology = inner_partition(dev, champion, market, outer_start)
        policy, inner_graph_audit = choose_inner(inner_train, inner_valid, inner_champion)
        graph = TemporalHeterogeneousGraph().fit(outer_train)
        one, two, outer_graph_audit = graph.predict(outer_valid)
        baseline = outer_champion.prob.to_numpy(float)
        confidence = outer_champion.confidence_signal.to_numpy(float)
        high = bool_series(outer_champion.high_conf).to_numpy(bool)
        selected = policy["selected"]
        candidate = baseline.copy() if selected["spec"] is None else architecture_probability(baseline, one, two, selected["spec"])
        positions = diagnostic.index[diagnostic.event_id.isin(set(outer_valid.event_id))]
        probability_map = dict(zip(outer_valid.event_id, candidate))
        diagnostic.loc[positions, "prob"] = diagnostic.loc[positions, "event_id"].map(probability_map)
        diagnostic.loc[positions, "v80_model"] = selected["name"]
        outer_diagnostics = {}
        for spec in ARCHITECTURES:
            probability = architecture_probability(baseline, one, two, spec)
            outer_diagnostics[spec["name"]] = metric(outer_valid, probability, confidence, high)
        base_metric = metric(outer_valid, baseline, confidence, high)
        candidate_metric = metric(outer_valid, candidate, confidence, high)
        evidence = outer_champion[[
            "event_id", "event_group_id", "event_time_utc", "market", "ticker",
            "source_family", "fold", "y", "fwd_ret_30m", "confidence_signal", "high_conf",
        ]].copy()
        evidence["baseline_prob"] = baseline
        evidence["graph_one_hop_prob"] = one
        evidence["graph_two_hop_prob"] = two
        evidence["candidate_prob"] = candidate
        evidence["v80_model"] = selected["name"]
        evidence_parts.append(evidence)
        audits.append({
            "market": market, "fold": fold,
            "inner_chronology": inner_chronology, "outer_chronology": outer_chronology,
            "policy": policy, "inner_graph_audit": inner_graph_audit,
            "outer_graph_audit": outer_graph_audit,
            "outer_architecture_diagnostics_evaluation_only": outer_diagnostics,
            "outer_baseline": base_metric, "outer_candidate": candidate_metric,
            "outer_auc_delta": candidate_metric["auc"] - base_metric["auc"],
            "outer_ba_delta": candidate_metric["balanced_accuracy"] - base_metric["balanced_accuracy"],
            "outer_net_delta": candidate_metric["all_trade_mean_signed_net"] - base_metric["all_trade_mean_signed_net"],
            "policy_locked_before_outer_evaluation": True,
            "outer_labels_used_for_selection": False,
        })
        print(
            f"[V80 GRAPH] market={market} fold={fold} selected={selected['name']} "
            f"inner_auc_delta={selected['auc_delta']:+.6f} outer_auc_delta={audits[-1]['outer_auc_delta']:+.6f}",
            flush=True,
        )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    require(evidence.event_id.is_unique, "V80 evidence repeats an event")
    require(np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic.confidence_signal.to_numpy(float)), "V80 changed V69 confidence")
    require(np.array_equal(champion.high_conf.to_numpy(bool), diagnostic.high_conf.to_numpy(bool)), "V80 changed V69 high-confidence mask")
    return diagnostic, evidence, audits


def paired_bootstrap(evidence: pd.DataFrame, draws: int) -> dict[str, Any]:
    work = evidence.copy()
    timestamp = pd.to_datetime(work.event_time_utc, utc=True)
    work["block"] = np.where(work.market.eq("US"), timestamp.dt.strftime("US|%Y-%m"), timestamp.dt.strftime("KR|%Y-%m-%d"))
    blocks = [part for _, part in work.groupby(["market", "fold", "block"], sort=True)]
    require(len(blocks) >= 12, "too few V80 bootstrap blocks")
    generator = np.random.default_rng(SEED)
    auc_delta = []
    ba_delta = []
    net_delta = []
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))])
        y = sample.y.to_numpy(int)
        if np.unique(y).size != 2:
            continue
        base = sample.baseline_prob.to_numpy(float)
        candidate = sample.candidate_prob.to_numpy(float)
        auc_delta.append(float(roc_auc_score(y, candidate) - roc_auc_score(y, base)))
        ba_delta.append(float(balanced_accuracy_score(y, candidate >= 0.5) - balanced_accuracy_score(y, base >= 0.5)))
        returns = sample.fwd_ret_30m.to_numpy(float)
        base_net = np.where(base >= 0.5, 1.0, -1.0) * returns - COST
        candidate_net = np.where(candidate >= 0.5, 1.0, -1.0) * returns - COST
        net_delta.append(float(candidate_net.mean() - base_net.mean()))
    require(len(auc_delta) >= int(0.90 * draws), "V80 bootstrap lost too many draws")

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, float)
        return {
            "effective_draws": len(array), "lower95": float(np.quantile(array, 0.025)),
            "median": float(np.median(array)), "upper95": float(np.quantile(array, 0.975)),
            "probability_gt_zero": float(np.mean(array > 0.0)),
        }

    return {
        "method": "paired market-time block bootstrap over inner-locked graph architecture",
        "seed": SEED, "requested_draws": draws, "blocks": len(blocks),
        "auc_delta": interval(auc_delta), "balanced_accuracy_delta": interval(ba_delta),
        "all_trade_net_delta": interval(net_delta),
    }


def evaluate(champion: pd.DataFrame, diagnostic: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    baseline_report = json.loads((V69_DIR / "DEV_ROBUSTNESS_REPORT.json").read_text(encoding="utf-8"))
    baseline_summary = baseline_report["selected"]
    diagnostic_original = diagnostic[list(champion.columns)].copy()
    candidate_summary = v44.summarize(diagnostic_original)
    canonical_baseline = controller.canonical_research_gate(baseline_summary)
    canonical_candidate = controller.canonical_research_gate(candidate_summary)
    require(canonical_baseline["total"] == canonical_candidate["total"] == 14, "controller canonical gate is not 14")
    require(baseline_summary["research_gate"] == canonical_baseline, "V69/controller gate mismatch")
    require(candidate_summary["research_gate"] == canonical_candidate, "V80/controller gate mismatch")
    base = metric(evidence, evidence.baseline_prob, evidence.confidence_signal, evidence.high_conf)
    candidate = metric(evidence, evidence.candidate_prob, evidence.confidence_signal, evidence.high_conf)
    bootstrap = paired_bootstrap(evidence, draws)
    fold_auc = np.asarray([audit["outer_auc_delta"] for audit in audits], float)
    fold_net = np.asarray([audit["outer_net_delta"] for audit in audits], float)
    base_metrics = baseline_summary["metrics"]
    candidate_metrics = candidate_summary["metrics"]
    confidence_exact = np.array_equal(champion.confidence_signal.to_numpy(float), diagnostic_original.confidence_signal.to_numpy(float)) and np.array_equal(champion.high_conf.to_numpy(bool), diagnostic_original.high_conf.to_numpy(bool))
    checks = {
        "all_six_market_folds_evaluated": len(audits) == 6,
        "outer_auc_delta_gt_0_003": candidate["auc"] - base["auc"] > 0.003,
        "outer_balanced_accuracy_delta_gt_0": candidate["balanced_accuracy"] - base["balanced_accuracy"] > 0.0,
        "outer_all_trade_net_delta_ge_0": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"] >= 0.0,
        "positive_outer_fold_auc_fraction_ge_2_of_3": float(np.mean(fold_auc > 0.0)) >= 2.0 / 3.0,
        "nonnegative_outer_fold_net_fraction_ge_2_of_3": float(np.mean(fold_net >= 0.0)) >= 2.0 / 3.0,
        "bootstrap_auc_delta_probability_gt_zero_ge_0_75": bootstrap["auc_delta"]["probability_gt_zero"] >= 0.75,
        "bootstrap_net_delta_probability_gt_zero_ge_0_65": bootstrap["all_trade_net_delta"]["probability_gt_zero"] >= 0.65,
        "full_overall_auc_delta_gt_0_001": candidate_metrics["auc"] - base_metrics["auc"] > 0.001,
        "full_overall_ba_delta_ge_minus_0_001": candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"] >= -0.001,
        "hc_accuracy_delta_ge_minus_0_005": candidate_metrics["highconf_accuracy"] - base_metrics["highconf_accuracy"] >= -0.005,
        "hc_net_delta_ge_minus_0_0005": candidate_metrics["strategy_mean_signed_net"] - base_metrics["strategy_mean_signed_net"] >= -0.0005,
        "v69_confidence_and_highconf_exact": confidence_exact,
        "strict_nested_chronology_all_folds": all(audit["outer_chronology"]["strict_35m_embargo"] and audit["inner_chronology"]["strict_35m_embargo"] and not audit["outer_labels_used_for_selection"] for audit in audits),
        "graph_message_passing_non_degenerate": all(audit["outer_graph_audit"]["hop_one_changed_nodes"] > 0 and audit["outer_graph_audit"]["hop_two_changed_nodes"] > 0 and not audit["outer_graph_audit"]["ticker_node"] and not audit["outer_graph_audit"]["text_node"] for audit in audits),
    }
    material_pass = bool(all(checks.values()))
    selected = diagnostic_original if material_pass else champion.copy()
    selected_summary = candidate_summary if material_pass else baseline_summary
    canonical_selected = controller.canonical_research_gate(selected_summary)
    require(canonical_selected["total"] == 14 and selected_summary["research_gate"] == canonical_selected, "selected/controller gate mismatch")
    if not material_pass:
        require(selected.equals(champion), "V80 fallback is not exact complete V69")
    return {
        "status": "MATERIAL_PASS" if material_pass else "MATERIAL_EXPERIMENT_FAIL_EXACT_V69_FALLBACK",
        "material_pass": material_pass, "material_checks": checks,
        "outer_baseline": base, "outer_candidate": candidate,
        "outer_auc_delta": candidate["auc"] - base["auc"],
        "outer_ba_delta": candidate["balanced_accuracy"] - base["balanced_accuracy"],
        "outer_net_delta": candidate["all_trade_mean_signed_net"] - base["all_trade_mean_signed_net"],
        "bootstrap": bootstrap,
        "baseline_summary": baseline_summary, "candidate_summary": candidate_summary,
        "selected_summary": selected_summary,
        "canonical_gate_audit": {"baseline": canonical_baseline, "candidate": canonical_candidate, "selected": canonical_selected, "reported_equals_controller_recomputed": True},
        "immutability": {
            "confidence_exact": confidence_exact,
            "baseline_confidence_sha256": array_sha256(champion.confidence_signal.to_numpy(float)),
            "candidate_confidence_sha256": array_sha256(diagnostic_original.confidence_signal.to_numpy(float)),
            "baseline_highconf_sha256": array_sha256(champion.high_conf.to_numpy(bool)),
            "candidate_highconf_sha256": array_sha256(diagnostic_original.high_conf.to_numpy(bool)),
        },
        "fallback": {"activated": not material_pass, "policy": "exact V69 entire frame" if not material_pass else None, "exact_frame_verified": bool(material_pass or selected.equals(champion))},
        "diagnostic_frame": diagnostic_original, "selected_frame": selected,
    }


def write_outputs(authority: dict[str, Any], champion: pd.DataFrame, evidence: pd.DataFrame, audits: list[dict[str, Any]], evaluation: dict[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in evaluation.items() if not key.endswith("_frame")}
    reports = {
        "MODEL_COMPARISON.json": {
            "version": "V80", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "models": {"V69_CHAMPION": evaluation["baseline_summary"], "V80_DIAGNOSTIC_GRAPH": evaluation["candidate_summary"], "V80_FAIL_CLOSED_SELECTED": evaluation["selected_summary"]},
            "evaluation": compact, "authority_audit": authority,
            "validation": "cut-local typed graph; relation-normalized 1/2-hop message passing; nested chronology; 35-minute embargo; outer evaluation-only",
            "seal_authorized": False,
        },
        "DEV_ROBUSTNESS_REPORT.json": {
            "version": "V80", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "opened_dev_only": True, "selected": evaluation["selected_summary"],
            "candidate": evaluation["candidate_summary"], "material_checks": evaluation["material_checks"],
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "fallback_is_exact_entire_v69_frame": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "SOURCE_TRANSFER_REPORT.json": {
            "version": "V80", "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "selected_by_source_family": evaluation["selected_summary"]["metrics"]["by_source_family"],
            "candidate_by_source_family": evaluation["candidate_summary"]["metrics"]["by_source_family"],
            "selected_by_market": evaluation["selected_summary"]["metrics"]["by_market"],
            "candidate_by_market": evaluation["candidate_summary"]["metrics"]["by_market"],
            "outer": {
                "auc_delta": evaluation["outer_auc_delta"],
                "balanced_accuracy_delta": evaluation["outer_ba_delta"],
                "net_delta": evaluation["outer_net_delta"],
                "bootstrap_auc_delta_probability_gt_zero": evaluation["bootstrap"]["auc_delta"]["probability_gt_zero"],
            },
            "fallback_is_exact_v69": not evaluation["material_pass"],
            "seal_authorized": False,
        },
        "TEMPORAL_HETEROGENEOUS_GRAPH_REPORT.json": {
            "version": "V80", "hypothesis": HYPOTHESIS,
            "node_types": EXPECTED_NODE_TYPES, "architectures": ARCHITECTURES,
            "node_shrink": NODE_SHRINK, "message_anchor": MESSAGE_ANCHOR,
            "ticker_node": False, "text_or_semantic_retrieval": False,
            "nested_fold_audits": audits, "evaluation": compact, "authority_audit": authority,
        },
        "CANONICAL_RESEARCH_GATE_AUDIT.json": {
            "version": "V80", "status": "MATCH", "canonical_check_count": 14,
            "reported": evaluation["selected_summary"]["research_gate"],
            "controller_recomputed": evaluation["canonical_gate_audit"]["selected"],
        },
        "RUN_STATUS.json": {
            "version": VERSION, "hypothesis": HYPOTHESIS, "status": evaluation["status"],
            "material_pass": evaluation["material_pass"], "champion_changed": evaluation["material_pass"],
            "phase": "ROBUST_SURVIVOR" if evaluation["selected_summary"]["research_gate"]["robust_survivor"] else "RESEARCH_FAIL",
            "seal_state": "UNOPENED", "seal_authorized": False, "completed_at": now(),
        },
    }
    for name, payload in reports.items():
        atomic_json(payload, out / name)
    atomic_csv(pd.DataFrame([{
        "market": audit["market"], "fold": audit["fold"],
        "selected_model": audit["policy"]["selected"]["name"],
        "outer_auc_delta": audit["outer_auc_delta"],
        "outer_ba_delta": audit["outer_ba_delta"],
        "outer_net_delta": audit["outer_net_delta"],
        "graph_nodes": audit["outer_graph_audit"]["node_count"],
        "graph_edges": audit["outer_graph_audit"]["edge_count"],
        "strict_35m_embargo": True, "outer_labels_used_for_selection": False,
    } for audit in audits]), out / "V80_GRAPH_FOLD_AUDIT.csv")
    atomic_csv(evidence, out / "V80_GRAPH_OUTER_EVIDENCE.csv.gz")
    atomic_csv(evaluation["diagnostic_frame"], out / "V80_DIAGNOSTIC_GRAPH_OOF.csv.gz")
    atomic_csv(evaluation["selected_frame"], out / "V80_FAIL_CLOSED_SELECTED_OOF.csv.gz")
    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json({"version": VERSION, "hypothesis": HYPOTHESIS, "authority_experiment_id": EXPECTED_V69_EXPERIMENT_ID, "files": files}, out / "ARTIFACT_MANIFEST.json")
    return {"output": str(out), "artifact_count": len(files) + 1, "manifest_sha256": sha256(out / "ARTIFACT_MANIFEST.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_limits.configure()
    authority = verify_authority()
    dev, champion = load_authorized()
    if args.audit_only:
        print(json.dumps(clean({
            "status": "AUDIT_OK", "hypothesis": HYPOTHESIS,
            "authority": authority, "dev_rows": len(dev), "v69_rows": len(champion),
            "node_types": EXPECTED_NODE_TYPES, "ticker_node": False,
            "text_or_semantic_retrieval": False, "direct_zero_hop_prior_candidate": False,
            "canonical_controller_gate_checks": 14, "output_written": False,
        }), ensure_ascii=False, indent=2))
        return
    diagnostic, evidence, audits = run_nested(dev, champion, smoke=args.smoke_test)
    evaluation = evaluate(champion, diagnostic, evidence, audits, 250 if args.smoke_test else BOOTSTRAP_DRAWS)
    non_noop = [trial for trial in audits[0]["policy"]["trials"] if trial["spec"] is not None]
    best_non_noop = max(non_noop, key=lambda trial: (trial["eligible"], trial["score"], trial["auc_delta"]))
    summary = {
        "status": "SMOKE_OK" if args.smoke_test else evaluation["status"],
        "hypothesis": HYPOTHESIS, "folds_executed": len(audits),
        "selected_models": [audit["policy"]["selected"]["name"] for audit in audits],
        "outer_auc_delta": evaluation["outer_auc_delta"],
        "outer_ba_delta": evaluation["outer_ba_delta"],
        "outer_net_delta": evaluation["outer_net_delta"],
        "material_pass": evaluation["material_pass"],
        "fallback_is_exact_v69": not evaluation["material_pass"],
        "canonical_gate": evaluation["selected_summary"]["research_gate"],
        "output_written": False,
    }
    if args.smoke_test:
        summary["raw_inner_selected"] = audits[0]["policy"]["selected"]
        summary["raw_inner_best_non_noop"] = best_non_noop
        summary["raw_outer_baseline"] = audits[0]["outer_baseline"]
        summary["raw_outer_selected"] = audits[0]["outer_candidate"]
        summary["raw_outer_best_non_noop_evaluation_only"] = audits[0]["outer_architecture_diagnostics_evaluation_only"][best_non_noop["name"]]
        summary["graph_audit"] = audits[0]["outer_graph_audit"]
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return
    result = write_outputs(authority, champion, evidence, audits, evaluation, args.output)
    print(json.dumps(clean({**summary, "output_written": True, "controller": result}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
