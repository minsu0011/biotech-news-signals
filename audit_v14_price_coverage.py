"""Label-blind current-version price timestamp coverage audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def audit_file(path: Path) -> dict[str, int]:
    events = pd.read_csv(path, dtype={"ticker": str})
    events["event_time_utc"] = pd.to_datetime(events.event_time_utc, utc=True)
    store = app.PriceStore(app.Config())
    counts = {
        "events": len(events),
        "ticker_file_missing": 0,
        "entry_or_exit_missing": 0,
        "crosses_session": 0,
        "prehistory_missing": 0,
        "structurally_usable": 0,
    }
    for event in events.itertuples(index=False):
        bars = store.load(str(event.market), str(event.ticker))
        if bars is None or bars.empty:
            counts["ticker_file_missing"] += 1
            continue
        event_time = pd.Timestamp(event.event_time_utc)
        local, _, _ = app.session_bounds(event_time, str(event.market))
        entry_target = event_time + pd.Timedelta(minutes=2)
        exit_target = entry_target + pd.Timedelta(minutes=30)
        values = bars.datetime.astype("int64").to_numpy()
        entry_index = int(np.searchsorted(values, int(entry_target.value)))
        exit_index = int(np.searchsorted(values, int(exit_target.value)))
        if entry_index >= len(values) or exit_index >= len(values) or exit_index <= entry_index:
            counts["entry_or_exit_missing"] += 1
            continue
        timezone = app.US_TZ if event.market == "US" else app.KR_TZ
        entry_time = bars.iloc[entry_index].datetime.tz_convert(timezone)
        exit_time = bars.iloc[exit_index].datetime.tz_convert(timezone)
        if entry_time.date() != local.date() or exit_time.date() != local.date():
            counts["crosses_session"] += 1
            continue
        if entry_index < 10:
            counts["prehistory_missing"] += 1
            continue
        counts["structurally_usable"] += 1
    return counts


def main() -> None:
    for name in (
        "events_sec_v17_dev.csv",
        "events_sec_v17_seal.csv",
        "events_kind_v17_dev.csv",
        "events_kind_v17_seal.csv",
    ):
        print(name, audit_file(DATA / name))


if __name__ == "__main__":
    main()
