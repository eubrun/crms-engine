import importlib.util
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


def snap(price=101, sar=100, b8=True, b12=True, bd=True, bw=True):
    return {"price": price, "quality_score": None,
            "8h": {"bull": b8}, "12h": {"bull": b12},
            "1d": {"bull": bd, "sar": sar, "bar": "2026-09-26"},
            "1w": {"bull": bw}}


class ScannerTest(unittest.TestCase):
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
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT", snap(bw=False), "t1"), ["BUY"])
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT", snap(bw=False), "t2"), [])
        self.assertEqual(scanner.update_symbol(data, "DOTUSDT", snap(b8=False, bw=False), "t3"), ["SELL"])
        self.assertEqual(len(data["trades"]), 1)
        self.assertEqual(data["trades"][0]["exit_tf"], "8h")

    def test_weekly_bull_profit_branches(self):
        for price, expected in [(103, "8h"), (106, "12h"), (112, "1d")]:
            data = {"version": 1, "symbols": {}, "trades": []}
            scanner.update_symbol(data, "SOLUSDT", snap(price=99), "t0")
            scanner.update_symbol(data, "SOLUSDT", snap(price=101), "t1")
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
