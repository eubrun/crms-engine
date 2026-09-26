"""Paper-trading scanner for all active Binance spot USDT pairs.

PSAR uses closed candles; the live price crosses the last closed Daily SAR.
State must reside on a mounted volume (CRMS_STATE_PATH) in production.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from crms import psar
from entry_quality import hourly_history, score as entry_score

API = os.getenv("CRMS_MARKET_API", "https://data-api.binance.vision/api/v3").rstrip("/")
PRIORITY = {"BCH", "LTC", "SOL", "SUI", "ICP", "DOT", "XRP", "AVAX",
            "XLM", "ZEC", "HYPE", "DYDX", "ONDO", "INJ", "RAY"}
STATE = Path(os.getenv("CRMS_STATE_PATH", "output/live_state.json"))
SCORE_PATH = Path(os.getenv("CRMS_SCORE_PATH", "/data/phase31_score.joblib"))
SESSION = requests.Session()


def exchange_universe():
    r = SESSION.get(f"{API}/exchangeInfo", timeout=30)
    r.raise_for_status()
    data = r.json()
    symbols = sorted({x["symbol"] for x in data["symbols"]
                      if x.get("status") == "TRADING" and x.get("quoteAsset") == "USDT"
                      and x.get("isSpotTradingAllowed", True)})
    if not symbols:
        raise RuntimeError("exchangeInfo returned no tradable USDT spot pairs")
    available = {s[:-4] for s in symbols}
    print("UNIVERSE|available=%d|priority_present=%s|priority_missing=%s" %
          (len(symbols), ",".join(sorted(PRIORITY & available)),
           ",".join(sorted(PRIORITY - available))), flush=True)
    return symbols


def klines(symbol, interval, limit=1000):
    r = SESSION.get(f"{API}/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError("no candles")
    columns = ["ot", "open", "high", "low", "close", "volume", "ct",
               "q", "n", "tb", "tq", "i"]
    d = pd.DataFrame(rows, columns=columns)
    for c in ["open", "high", "low", "close", "volume", "ct"]:
        d[c] = pd.to_numeric(d[c])
    d["date"] = pd.to_datetime(d.ot, unit="ms", utc=True)
    return d.set_index("date")


def market_state(raw, now_ms, minimum=3):
    closed = raw[raw.ct < now_ms][["open", "high", "low", "close", "volume"]]
    if len(closed) < minimum:
        raise ValueError("fewer than %d closed candles" % minimum)
    p = psar(closed)
    return {"bull": bool(p.bull.iloc[-1]), "sar": float(p.psar.iloc[-1]),
            "bar": closed.index[-1].isoformat(), "closed": closed}


def load_state():
    if not STATE.exists():
        return {"version": 1, "symbols": {}, "trades": []}
    data = json.loads(STATE.read_text())
    if data.get("version") != 1:
        raise RuntimeError("unknown state version")
    return data


def save_state(data):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(STATE.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, STATE)


def update_symbol(data, symbol, snapshot, timestamp):
    """Return BUY/SELL events; replaying the same snapshot does not duplicate trades."""
    rows = data["symbols"]
    previous = rows.get(symbol)
    daily = snapshot["1d"]
    price = snapshot["price"]
    bullish = price > daily["sar"]
    current = {"above_daily": bullish, "daily_bar": daily["bar"],
               "sar_daily": daily["sar"], "price": price,
               "bull_8h": snapshot["8h"]["bull"],
               "bull_12h": snapshot["12h"]["bull"],
               "bull_daily": daily["bull"]}
    events = []
    open_trade = next((t for t in reversed(data["trades"])
                       if t["symbol"] == symbol and t["status"] == "OPEN"), None)
    if previous is not None and not previous["above_daily"] and bullish and open_trade is None:
        trade = {"symbol": symbol, "status": "OPEN", "entered_at": timestamp,
                 "entry": price, "weekly_at_entry": snapshot["1w"]["bull"],
                 "quality_score": snapshot.get("quality_score"), "max_price": price,
                 "first_8h_bear": None, "exit_tf": None}
        data["trades"].append(trade)
        open_trade = trade
        events.append("BUY")
    if open_trade is not None:
        open_trade["max_price"] = max(open_trade["max_price"], price)
        flip8 = previous is not None and previous["bull_8h"] and not snapshot["8h"]["bull"]
        flip12 = previous is not None and previous["bull_12h"] and not snapshot["12h"]["bull"]
        flip_daily = previous is not None and previous["bull_daily"] and not daily["bull"]
        if open_trade["first_8h_bear"] is None and flip8:
            pnl = price / open_trade["entry"] - 1
            open_trade["first_8h_bear"] = {"at": timestamp, "pnl": pnl}
            if not open_trade["weekly_at_entry"] or pnl < .04:
                open_trade["exit_tf"] = "8h"
            elif pnl < .10:
                open_trade["exit_tf"] = "12h"
            else:
                open_trade["exit_tf"] = "1d"
        tf = open_trade["exit_tf"]
        if tf and ((tf == "8h" and flip8) or (tf == "12h" and flip12)
                   or (tf == "1d" and flip_daily)):
            open_trade.update(status="CLOSED", exited_at=timestamp, exit=price,
                              return_pct=100 * (price / open_trade["entry"] - 1))
            events.append("SELL")
    rows[symbol] = current
    return events


def scan(data, symbols):
    now_ms = int(time.time() * 1000)
    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    failures = {}
    for symbol in symbols:
        try:
            snapshots = {}
            for tf in ("8h", "12h", "1d", "1w"):
                raw = klines(symbol, tf)
                snapshots[tf] = market_state(raw, now_ms)
                if tf == "1d":
                    snapshots["price"] = float(raw.close.iloc[-1])
            snapshots["quality_score"] = None
            near_cross = abs(snapshots["price"] / snapshots["1d"]["sar"] - 1) <= .05
            prior = data["symbols"].get(symbol)
            new_cross = (prior is not None and not prior["above_daily"]
                         and snapshots["price"] > snapshots["1d"]["sar"])
            if SCORE_PATH.exists() and (near_cross or new_cross):
                try:
                    history = hourly_history(SESSION, API, symbol, now_ms)
                    snapshots["quality_score"] = entry_score(
                        SCORE_PATH, history, snapshots["price"])
                except Exception as score_exc:
                    print("SCOREFAIL|%s|%s" % (symbol, str(score_exc)[:160]), flush=True)
            events = update_symbol(data, symbol, snapshots, timestamp)
            save_state(data)
            d = snapshots["1d"]
            quality = snapshots["quality_score"]
            band = "NA" if quality is None else ("HIGH" if quality >= 80 else
                                                  "MEDIUM" if quality >= 60 else "LOW")
            print("LIVE|%s|px=%.8g|sarD=%.8g|8H=%d|12H=%d|D=%d|W=%d|score=%s|quality=%s|%s" %
                  (symbol, snapshots["price"], d["sar"], snapshots["8h"]["bull"],
                   snapshots["12h"]["bull"], d["bull"], snapshots["1w"]["bull"],
                   "NA" if quality is None else str(quality), band,
                   ",".join(events) or "WAIT"), flush=True)
        except Exception as exc:
            failures[symbol] = str(exc)[:160]
            print("LIVEFAIL|%s|%s" % (symbol, failures[symbol]), flush=True)
    print("SCAN|DONE|scanned=%d|failed=%d" % (len(symbols) - len(failures), len(failures)),
          flush=True)


def main():
    data = load_state()
    print("LIVE|START|state=%s" % STATE, flush=True)
    while True:
        try:
            scan(data, exchange_universe())
        except Exception as exc:
            print("SCANFAIL|%s" % exc, flush=True)
        time.sleep(3600)


if __name__ == "__main__":
    main()
