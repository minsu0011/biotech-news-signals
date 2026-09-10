"""Metadata-only pre-freeze integrity audit for the untouched pure V35 seal."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEAL = {("US", "US_EXACT_V35_SEAL"), ("KR", "KR_EXACT_V35_SEAL")}
DEV: set[tuple[str, str]] = set()


def pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame.market.astype(str), frame.source.astype(str)))


def main() -> None:
    cfg = app.Config()
    source_status = json.loads((DATA / "V35_SOURCE_STATUS.json").read_text(encoding="utf-8"))
    if source_status.get("selection_uses_labels") is not False:
        raise RuntimeError("V35 source selection was not declared label blind")
    for key in ("prior_event_id_overlap", "prior_article_overlap", "prior_url_overlap"):
        if int(source_status.get(key, -1)) != 0:
            raise RuntimeError(f"source leakage: {key}={source_status.get(key)}")

    events = pd.read_csv(
        DATA / "events_exact_v35.csv.gz", compression="gzip", dtype={"ticker": str},
        usecols=lambda column: column in {
            "event_id", "market", "ticker", "source", "event_time_utc", "article_id",
        },
    )
    events["event_time_utc"] = pd.to_datetime(events.event_time_utc, utc=True)
    metadata = pd.read_csv(
        DATA / "labeled_events_v35.csv.gz", compression="gzip", dtype={"ticker": str},
        usecols=["event_id", "market", "ticker", "source", "event_time_utc"],
    )
    metadata["event_time_utc"] = pd.to_datetime(metadata.event_time_utc, utc=True)
    search = app.load_search_frame(cfg)
    seal_mask = pd.Series([
        (str(market), str(source)) in SEAL
        for market, source in zip(search.market, search.source)
    ], index=search.index)
    if not seal_mask.any() or search.loc[seal_mask, "y"].notna().any():
        raise RuntimeError("V35 sealed labels were exposed or seal is empty")
    if search.loc[seal_mask, "fwd_ret_30m"].notna().any():
        raise RuntimeError("V35 sealed forward returns were exposed")
    current_events = events[[
        (str(market), str(source)) in SEAL | DEV
        for market, source in zip(events.market, events.source)
    ]].copy()
    current_metadata = metadata[[
        (str(market), str(source)) in SEAL | DEV
        for market, source in zip(metadata.market, metadata.source)
    ]].copy()
    prior = pd.read_csv(
        DATA / "events_exact_v34.csv.gz", compression="gzip", usecols=["event_id"]
    )
    overlap = set(current_events.event_id.astype(str)) & set(prior.event_id.astype(str))
    if overlap:
        raise RuntimeError(f"V35/V34 event overlap: {len(overlap)}")
    if current_events.event_id.duplicated().any() or current_metadata.event_id.duplicated().any():
        raise RuntimeError("duplicate current event IDs")
    if not SEAL.issubset(pairs(current_metadata)):
        raise RuntimeError("one or more fixed V35 seal partitions have no usable labeled metadata")

    event_counts = current_events.groupby(["market", "source"]).size().to_dict()
    labeled_counts = current_metadata.groupby(["market", "source"]).size().to_dict()
    market_counts = current_metadata.groupby("market").size().to_dict()
    if (len(current_metadata) < cfg.cert_min_total_events
            or int(market_counts.get("US", 0)) < cfg.cert_min_us_events
            or int(market_counts.get("KR", 0)) < cfg.cert_min_kr_events):
        raise RuntimeError(f"V35 certification sample minimum failed: {market_counts}")
    report = {
        "status": "PASS",
        "v35_seal_outcomes_loaded": False,
        "selection_uses_labels": False,
        "prior_event_id_overlap": len(overlap),
        "event_counts": {f"{market}|{source}": int(count)
                         for (market, source), count in event_counts.items()},
        "usable_labeled_metadata_counts": {
            f"{market}|{source}": int(count)
            for (market, source), count in labeled_counts.items()
        },
        "usable_total": int(len(current_metadata)),
        "search_frame_seal_metadata_rows": int(seal_mask.sum()),
        "search_frame_seal_y_nonnull": int(search.loc[seal_mask, "y"].notna().sum()),
        "search_frame_seal_return_nonnull": int(
            search.loc[seal_mask, "fwd_ret_30m"].notna().sum()
        ),
    }
    path = ROOT / "output" / "V35" / "PRESEAL_INTEGRITY.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
