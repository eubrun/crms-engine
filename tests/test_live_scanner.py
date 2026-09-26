import importlib.util
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "sar_live_scanner", Path(__file__).resolve().parents[1] / "sar_live_scanner.py")
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


def snap(price=101, sar=100, b8=True, b12=True, bd=True, bw=True, cross=False):
    bears = {tf: ([{"bar": "2026-09-26T08:00:00+00:00",
                    "at": "2026-09-26T08:01:00+00:00", "price": price,
                    "trigger_sar": price}] if triggered else [])
             for tf, triggered in (("8h", not b8), ("12h", not b12), ("1d", not bd))}
    return {"price": price, "quality_score": None,
            "bear_crosses": bears,
            "daily_cross": ({"day": "2026-09-26", "at": "2026-09-26T07:01:00+00:00",
                             "price": 100., "trigger_sar": sar} if cross else None),
            "8h": {"bull": b8}, "12h": {"bull": b12},
            "1d": {"bull": bd, "sar": sar, "bar": "2026-09-26"},
            "1w": {"bull": bw}}


class ScannerTest(unittest.TestCase):
    def test_report_separates_cross_from_scan_price_and_live_sar(self):
        snapshot = snap(price=99, sar=100, cross=True, b8=False)
        snapshot.update(daily_high=102, daily_live_bull=True, daily_live_sar=95)
        for tf, sar, live in (("8h", 101, 107), ("12h", 102, 108),
                              ("1d", 100, 95)):
            snapshot[tf].update(sar=sar, live_sar=live, live_bull=True)
        line = scanner.live_log_line("DOTUSDT", snapshot, ["BUY"])
        for field in ("px=99", "crossPx=100", "crossSAR=100", "sarLiveD=95",
                      "bear8Px=99", "bear8SAR=99", "sar8=101", "sarLive8=107"):
            self.assertIn("|" + field + "|", line + "|")

    def test_high_cross_is_kept_when_scan_price_falls_back(self):
        data = {"version": 1, "symbols": {}, "trades": []}
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT",
                         snap(price=99, cross=True, bw=False), "t1"), ["BUY"])
        self.assertEqual(data["trades"][0]["entry"], 100.)
        self.assertEqual(data["trades"][0]["detected_at"], "t1")
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT",
                         snap(price=98, cross=True, bw=False), "t2"), [])
        self.assertEqual(len(data["trades"]), 1)

    def test_first_cross_minute_is_reconstructed_from_daily_high(self):
        day = pd.Timestamp("2026-09-26", tz="UTC")
        daily = pd.DataFrame({
            "open": [95., 96.], "high": [98., 102.], "low": [94., 97.],
            "close": [97., 99.], "volume": [1., 1.],
            "ct": [int((day-pd.Timedelta(milliseconds=1)).timestamp()*1000),
                   int((day+pd.Timedelta(days=1)-pd.Timedelta(milliseconds=1)).timestamp()*1000)]
        }, index=[day-pd.Timedelta(days=1), day])
        minute = pd.DataFrame({
            "open": [98., 99., 101.], "high": [99., 101., 102.],
            "low": [97., 98., 99.], "close": [98., 100., 99.]
        }, index=pd.date_range(day, periods=3, freq="min"))
        with patch.object(scanner, "klines", return_value=minute):
            event = scanner.first_daily_cross(
                "DOTUSDT", daily, {"bull": False, "sar": 100.},
                int((day+pd.Timedelta(hours=8)).timestamp()*1000))
        self.assertEqual(event["at"], (day+pd.Timedelta(minutes=1)).isoformat())
        self.assertEqual(event["price"], 100.)

    def test_bearish_candle_low_is_kept_after_price_recovers(self):
        day = pd.Timestamp("2026-09-26", tz="UTC")
        idx = pd.date_range(day - pd.Timedelta(hours=24), periods=4, freq="8h")
        raw = pd.DataFrame({"open": [105.] * 4, "high": [110.] * 4,
                            "low": [103., 103., 103., 99.],
                            "close": [106., 106., 106., 106.],
                            "volume": [1.] * 4,
                            "ct": [int((d + pd.Timedelta(hours=8) -
                                        pd.Timedelta(milliseconds=1)).timestamp()*1000)
                                   for d in idx]}, index=idx)
        minute = pd.DataFrame({"open": [105., 101., 99.],
                               "high": [106., 102., 107.],
                               "low": [104., 99., 98.]},
                              index=pd.date_range(idx[-1], periods=3, freq="min"))
        class PsarResult:
            bull = pd.Series([True])
            psar = pd.Series([100.])
        with patch.object(scanner, "psar", return_value=PsarResult()), \
             patch.object(scanner, "klines", return_value=minute):
            events = scanner.bearish_crosses("DOTUSDT", raw,
                int((idx[-1] + pd.Timedelta(hours=1)).timestamp()*1000))
        self.assertEqual(events[-1]["price"], 100.)
        self.assertEqual(events[-1]["at"], minute.index[1].isoformat())

    def test_rome_schedule_and_daylight_saving(self):
        slot = scanner.next_scan_slot(datetime(2026, 9, 26, 7, 58, tzinfo=timezone.utc))
        self.assertEqual(slot.isoformat(), "2026-09-26T10:00:00+02:00")
        self.assertEqual(scanner.next_scan_slot(
            datetime(2026, 9, 26, 8, 1, tzinfo=timezone.utc), slot).hour, 13)
        winter = scanner.next_scan_slot(
            datetime(2026, 12, 1, 8, 58, tzinfo=timezone.utc))
        self.assertEqual(winter.isoformat(), "2026-12-01T10:00:00+01:00")
        following = scanner.next_scan_slot(
            datetime(2026, 12, 1, 16, 3, tzinfo=timezone.utc))
        self.assertEqual(following.isoformat(), "2026-12-02T10:00:00+01:00")

    def test_dynamic_universe_excludes_known_non_crypto(self):
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"symbols": [
                    {"symbol": "HYPEUSDT", "baseAsset": "HYPE", "quoteAsset": "USDT",
                     "status": "TRADING"},
                    {"symbol": "DOTUSDT", "baseAsset": "DOT", "quoteAsset": "USDT",
                     "status": "TRADING"},
                    {"symbol": "FDUSDUSDT", "baseAsset": "FDUSD", "quoteAsset": "USDT",
                     "status": "TRADING"},
                    {"symbol": "AAPLBUSDT", "baseAsset": "AAPLB", "quoteAsset": "USDT",
                     "status": "TRADING"}]}
        with patch.object(scanner.SESSION, "get", return_value=Response()):
            self.assertEqual(scanner.exchange_universe(), ["DOTUSDT", "HYPEUSDT"])

    def test_phase31_feature_schema(self):
        from entry_quality import FEATURES, live_features
        index = pd.date_range("2026-01-01", periods=24*220, freq="h", tz="UTC")
        values = np.linspace(90, 110, len(index))
        history = pd.DataFrame({"open": values, "high": values+1,
                                "low": values-1, "close": values,
                                "volume": 100.}, index=index)
        features = live_features(history, 110.)
        self.assertEqual(list(features.columns), FEATURES)
        self.assertEqual(len(FEATURES), 31)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_weekly_bear_exit_and_no_duplicate_buy(self):
        data = {"version": 1, "symbols": {}, "trades": []}
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT", snap(price=99, bw=False), "t0"), [])
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT", snap(bw=False, cross=True), "t1"), ["BUY"])
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT", snap(bw=False), "t2"), [])
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT", snap(b8=False, bw=False), "t3"), ["SELL"])
        self.assertEqual(len(data["trades"]), 1)
        self.assertEqual(data["trades"][0]["exit_tf"], "8h")

    def test_weekly_bull_profit_branches(self):
        for price, expected in [(103, "8h"), (106, "12h"), (112, "1d")]:
            data = {"version": 1, "symbols": {}, "trades": []}
            scanner.update_symbol(data, "SOLUSDT", snap(price=99), "t0")
            scanner.update_symbol(data, "SOLUSDT", snap(price=101, cross=True), "t1")
            scanner.update_symbol(data, "SOLUSDT", snap(price=price, b8=False), "t2")
            trade = data["trades"][0]
            self.assertEqual(trade["exit_tf"], expected)
            self.assertEqual(trade["status"], "CLOSED" if expected == "8h" else "OPEN")
            if expected == "12h":
                self.assertEqual(scanner.update_symbol(data, "SOLUSDT",
                    snap(price=price, b8=False, b12=False), "t3"), ["SELL"])
            if expected == "1d":
                self.assertEqual(scanner.update_symbol(data, "SOLUSDT",
                    snap(price=price, b8=False, b12=False, bd=False), "t3"), ["SELL"])

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            old = scanner.STATE
            scanner.STATE = Path(directory) / "state.json"
            try:
                state = {"version": 1, "symbols": {"X": {"above_daily": False}}, "trades": []}
                scanner.save_state(state)
                self.assertEqual(scanner.load_state(), state)
            finally:
                scanner.STATE = old


if __name__ == "__main__":
    unittest.main()
