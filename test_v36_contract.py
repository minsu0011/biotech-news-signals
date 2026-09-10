"""Unit tests for the V36 exact t+2 / actual-entry+30m contract."""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

import bio_news_30m_v3 as app


def minute_bars() -> pd.DataFrame:
    timestamp=pd.date_range("2026-08-24 13:30:00+00:00",periods=391,freq="1min")
    value=100.0+np.arange(len(timestamp),dtype=float)*0.01
    bars=pd.DataFrame({
        "datetime":timestamp,
        "open":value,
        "high":value+0.02,
        "low":value-0.02,
        "close":value+0.01,
        "volume":1000.0+np.arange(len(timestamp),dtype=float),
    })
    bars.attrs.update({
        "price_source":"SYNTHETIC_AUDITED_1M",
        "price_cadence_sec":60,
        "bar_timestamp_semantics":"OPEN",
        "official_exact_eligible":True,
    })
    return bars


def event() -> pd.Series:
    return pd.Series({
        "event_id":"SEC:1:unit-test",
        "market":"US",
        "ticker":"TEST",
        "event_time_utc":"2026-08-24 14:00:30+00:00",
        "source":"SEC_V36_UNIT",
        "form":"8-K",
        "event_type":"REGULATORY",
        "headline":"Clinical result unit test",
        "body":"",
        "url":"https://example.invalid/unit",
        "article_id":"SEC:1:unit-test",
    })


class ExactContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg=replace(app.Config(),official_exact_contract=True)

    def test_entry_exit_are_open_stamped_and_actual_entry_anchored(self) -> None:
        row,status=app.event_to_row(event(),minute_bars(),self.cfg)
        self.assertEqual(status,"OK")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["entry_target_utc"],pd.Timestamp("2026-08-24 14:02:30+00:00"))
        self.assertEqual(row["actual_entry_time_utc"],pd.Timestamp("2026-08-24 14:03:00+00:00"))
        self.assertEqual(row["entry_slippage_sec"],30.0)
        self.assertEqual(row["exit_target_utc"],pd.Timestamp("2026-08-24 14:33:00+00:00"))
        self.assertEqual(row["actual_exit_time_utc"],pd.Timestamp("2026-08-24 14:33:00+00:00"))
        self.assertEqual(row["actual_hold_seconds"],1800.0)
        bars=minute_bars()
        expected=float(bars.loc[bars.datetime.eq(row["actual_entry_time_utc"]),"open"].iloc[0])
        self.assertEqual(row["entry_price"],expected)

    def test_five_minute_proxy_is_rejected(self) -> None:
        bars=minute_bars()
        bars.attrs.update({"price_cadence_sec":300,"official_exact_eligible":False})
        row,status=app.event_to_row(event(),bars,self.cfg)
        self.assertIsNone(row)
        self.assertEqual(status,"NOT_EXACT_PRICE_ELIGIBLE")

    def test_entry_timing_miss_is_not_silently_accepted(self) -> None:
        bars=minute_bars()
        bars=bars[~bars.datetime.isin(pd.to_datetime([
            "2026-08-24 14:03:00+00:00","2026-08-24 14:04:00+00:00"
        ]))].reset_index(drop=True)
        bars.attrs.update(minute_bars().attrs)
        row,status=app.event_to_row(event(),bars,self.cfg)
        self.assertIsNone(row)
        self.assertEqual(status,"ENTRY_TIMING_MISS")

    def test_exit_timing_miss_is_not_silently_accepted(self) -> None:
        bars=minute_bars()
        bars=bars[~bars.datetime.isin(pd.to_datetime([
            "2026-08-24 14:33:00+00:00","2026-08-24 14:34:00+00:00"
        ]))].reset_index(drop=True)
        bars.attrs.update(minute_bars().attrs)
        row,status=app.event_to_row(event(),bars,self.cfg)
        self.assertIsNone(row)
        self.assertEqual(status,"EXIT_TIMING_MISS")

    def test_features_ignore_future_bar_contents(self) -> None:
        original=minute_bars()
        row_a,status_a=app.event_to_row(event(),original,self.cfg)
        self.assertEqual(status_a,"OK")
        mutated=minute_bars()
        future=mutated.datetime>=pd.Timestamp("2026-08-24 14:03:00+00:00")
        mutated.loc[future,["high","low","close","volume"]]=1_000_000.0
        mutated.attrs.update(original.attrs)
        row_b,status_b=app.event_to_row(event(),mutated,self.cfg)
        self.assertEqual(status_b,"OK")
        assert row_a is not None and row_b is not None
        causal=[name for name in app.NUMERIC_COLS if not name.startswith("benchmark_") and
                not name.startswith("excess_ret_")]
        for name in causal:
            a=row_a.get(name,np.nan);b=row_b.get(name,np.nan)
            if pd.isna(a) and pd.isna(b):
                continue
            self.assertAlmostEqual(float(a),float(b),places=12,msg=name)

    def test_source_family_and_group_are_deterministic(self) -> None:
        ev=event()
        self.assertEqual(app.source_family_for_event(ev),"US_SEC")
        self.assertEqual(app.canonical_event_group_id(ev),app.canonical_event_group_id(ev.copy()))


if __name__=="__main__":
    unittest.main(verbosity=2)
