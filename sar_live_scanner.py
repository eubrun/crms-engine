"""Paper-trading scanner for active Binance spot USDT pairs.

Intrabar highs/lows cross the previous closed SAR; minute candles date paper events.
State must reside on a mounted volume (CRMS_STATE_PATH) in production.
"""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from crms import psar
from entry_quality import hourly_history, score as entry_score

API = os.getenv("CRMS_MARKET_API", "https://data-api.binance.vision/api/v3").rstrip("/")
PRIORITY = {"BCH", "LTC", "SOL", "SUI", "ICP", "DOT", "XRP", "AVAX",
            "XLM", "ZEC", "HYPE", "DYDX", "ONDO", "INJ", "RAY"}
NON_CRYPTO_BASES = {
    "EUR", "EURI", "GBP", "AUD", "BRL", "TRY", "ARS", "UAH", "BIDR", "IDRT",
    "USDC", "FDUSD", "BFUSD", "TUSD", "USDP", "USD1", "USDE", "PYUSD", "DAI",
    "AAOIB", "AAPLB", "AMZNB", "AVGOB", "ASMLB", "ASTSB", "CRCLB", "CRDOB",
    "DELLB", "EWYB", "FLNCB", "GLWB", "GMEB", "GOOGLB", "GSB", "HOODB",
    "IBMB", "MSFTB", "NVDAB", "TSLAB",
}
STATE = Path(os.getenv("CRMS_STATE_PATH", "output/live_state.json"))
SCORE_PATH = Path(os.getenv("CRMS_SCORE_PATH", "/data/phase31_score.joblib"))
SESSION = requests.Session()
ROME = ZoneInfo("Europe/Rome")
SCAN_HOURS = (10, 13, 17)


def exchange_universe():
    r = SESSION.get(f"{API}/exchangeInfo", timeout=30)
    r.raise_for_status()
    data = r.json()
    symbols = sorted({x["symbol"] for x in data["symbols"]
                      if x.get("status") == "TRADING" and x.get("quoteAsset") == "USDT"
                      and x.get("isSpotTradingAllowed", True)
                      and x.get("baseAsset") not in NON_CRYPTO_BASES},
                     key=lambda s: (s[:-4] not in PRIORITY, s))
    if not symbols:
        raise RuntimeError("exchangeInfo returned no tradable USDT spot pairs")
    available = {s[:-4] for s in symbols}
    print("UNIVERSE|available=%d|priority_present=%s|priority_missing=%s" %
          (len(symbols), ",".join(sorted(PRIORITY & available)),
           ",".join(sorted(PRIORITY - available))), flush=True)
    return symbols


def klines(symbol, interval, limit=1000, start_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    r = SESSION.get(f"{API}/klines",
                    params=params,
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


def minute_history(symbol, start, end):
    """Read an entire candle, including Daily candles longer than 1,000 minutes."""
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    parts = []
    while cursor <= end_ms:
        part = klines(symbol, "1m", start_time=cursor)
        part = part[part.index <= end]
        if part.empty:
            break
        parts.append(part)
        cursor = int(part.index[-1].timestamp() * 1000) + 60_000
        if len(part) < 1000:
            break
    return pd.concat(parts) if parts else pd.DataFrame()


def first_daily_cross(symbol, daily_raw, daily_state, now_ms):
    """Find the first minute of today's high crossing the prior closed Daily SAR.

    A high proves the threshold was traded even if the price later falls back.
    The event price is theoretical, not an executable quote at scan time.
    """
    if daily_state["bull"]:
        return None
    current = daily_raw.iloc[-1]
    if current.ct < now_ms or float(current.high) <= daily_state["sar"]:
        return None
    minute = minute_history(symbol, daily_raw.index[-1],
                            pd.Timestamp(now_ms, unit="ms", tz="UTC"))
    minute = minute[(minute.index >= daily_raw.index[-1]) & (minute.index <=
             pd.Timestamp(now_ms, unit="ms", tz="UTC"))]
    crossed = minute[minute.high > daily_state["sar"]]
    if crossed.empty:
        raise ValueError("daily high crossed SAR but minute history did not confirm it")
    first = crossed.iloc[0]
    return {"day": daily_raw.index[-1].date().isoformat(),
            "at": crossed.index[0].isoformat(),
            "price": max(float(first.open), daily_state["sar"]),
            "trigger_sar": daily_state["sar"]}


def bearish_crosses(symbol, raw, now_ms, bars=4):
    """Reconstruct bearish SAR touches within recent candles from one-minute lows."""
    closed = raw[raw.ct < now_ms][["open", "high", "low", "close", "volume"]]
    if len(closed) < 3:
        return []
    first = max(2, len(closed) - bars)
    candidates = []
    for i in range(first, len(raw)):
        bar = raw.iloc[i]
        if raw.index[i].timestamp() * 1000 > now_ms:
            continue
        prior = psar(raw.iloc[:i][["open", "high", "low", "close", "volume"]])
        if not bool(prior.bull.iloc[-1]):
            continue
        threshold = float(prior.psar.iloc[-1])
        if float(bar.low) >= threshold:
            continue
        start = raw.index[i]
        end = min(pd.Timestamp(int(bar.ct), unit="ms", tz="UTC"),
                  pd.Timestamp(now_ms, unit="ms", tz="UTC"))
        minute = minute_history(symbol, start, end)
        minute = minute[(minute.index >= start) & (minute.index <= end)]
        crossed = minute[minute.low < threshold]
        if crossed.empty:
            raise ValueError("candle low crossed SAR but minute history did not confirm it")
        first_minute = crossed.iloc[0]
        candidates.append({"bar": start.isoformat(), "at": crossed.index[0].isoformat(),
                           "price": min(float(first_minute.open), threshold),
                           "trigger_sar": threshold})
    return candidates


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
    cross = snapshot.get("daily_cross")
    current = {"above_daily": bullish, "daily_bar": daily["bar"],
               "sar_daily": daily["sar"], "price": price,
               "last_cross_day": previous.get("last_cross_day") if previous else None,
               "last_bear_bar": dict(previous.get("last_bear_bar", {})) if previous else {},
               "bull_8h": snapshot["8h"]["bull"],
               "bull_12h": snapshot["12h"]["bull"],
               "bull_daily": daily["bull"]}
    events = []
    open_trade = next((t for t in reversed(data["trades"])
                       if t["symbol"] == symbol and t["status"] == "OPEN"), None)
    fresh_cross = cross is not None and current["last_cross_day"] != cross["day"]
    if fresh_cross:
        current["last_cross_day"] = cross["day"]
    if fresh_cross and open_trade is None:
        trade = {"symbol": symbol, "status": "OPEN", "entered_at": cross["at"],
                 "detected_at": timestamp, "entry": cross["price"],
                 "trigger_sar": cross["trigger_sar"],
                 "weekly_at_entry": snapshot["1w"]["bull"],
                 "quality_score": snapshot.get("quality_score"),
                 "max_price": max(price, cross["price"]),
                 "first_8h_bear": None, "exit_tf": None}
        data["trades"].append(trade)
        open_trade = trade
        events.append("BUY")
    if open_trade is not None:
        open_trade["max_price"] = max(open_trade["max_price"], price)
        bear_events = {}
        for tf in ("8h", "12h", "1d"):
            seen = current["last_bear_bar"].get(tf)
            fresh = [e for e in snapshot.get("bear_crosses", {}).get(tf, [])
                     if (seen is None or e["bar"] > seen)
                     and e["at"] >= open_trade["entered_at"]]
            bear_events[tf] = fresh[0] if fresh else None
        first8 = bear_events["8h"]
        if open_trade["first_8h_bear"] is None and first8:
            pnl = first8["price"] / open_trade["entry"] - 1
            open_trade["first_8h_bear"] = {"at": first8["at"], "pnl": pnl}
            if not open_trade["weekly_at_entry"] or pnl < .04:
                open_trade["exit_tf"] = "8h"
            elif pnl < .10:
                open_trade["exit_tf"] = "12h"
            else:
                open_trade["exit_tf"] = "1d"
        tf = open_trade["exit_tf"]
        event = bear_events.get(tf)
        if event and (tf == "8h" or (open_trade["first_8h_bear"] and
                                      event["at"] >= open_trade["first_8h_bear"]["at"])):
            open_trade.update(status="CLOSED", exited_at=event["at"],
                              exit=event["price"], exit_detected_at=timestamp,
                              exit_trigger_sar=event["trigger_sar"],
                              return_pct=100 * (event["price"] / open_trade["entry"] - 1))
            events.append("SELL")
    for tf in ("8h", "12h", "1d"):
        found = snapshot.get("bear_crosses", {}).get(tf, [])
        if found:
            current["last_bear_bar"][tf] = found[-1]["bar"]
    rows[symbol] = current
    return events


def live_log_line(symbol, snapshot, events):
    """Keep event, scan quote and provisional SAR separate in each report row."""
    fmt = lambda value: "%.8g" % value
    d = snapshot["1d"]
    quality = snapshot["quality_score"]
    band = "NA" if quality is None else ("HIGH" if quality >= 80 else
                                          "MEDIUM" if quality >= 60 else "LOW")
    fields = ["LIVE", symbol, "px=" + fmt(snapshot["price"]),
              "highD=" + fmt(snapshot["daily_high"]),
              "sarD=" + fmt(d["sar"]),
              "Dlive=%d" % snapshot["daily_live_bull"],
              "sarLiveD=" + fmt(snapshot["daily_live_sar"])]
    cross = snapshot["daily_cross"]
    fields.extend(["crossAt=" + (cross["at"] if cross else "NA"),
                   "crossPx=" + (fmt(cross["price"]) if cross else "NA"),
                   "crossSAR=" + (fmt(cross["trigger_sar"]) if cross else "NA")])
    for tf, label in (("8h", "8"), ("12h", "12"), ("1d", "D")):
        state = snapshot[tf]
        event_list = snapshot["bear_crosses"][tf]
        event = event_list[-1] if event_list else None
        fields.extend(["sar" + label + "=" + fmt(state["sar"]),
                       "sarLive" + label + "=" + fmt(state["live_sar"]),
                       "live" + label + "=%d" % state["live_bull"],
                       "bear" + label + "=" + (event["at"] if event else "NA"),
                       "bear" + label + "Px=" + (fmt(event["price"]) if event else "NA"),
                       "bear" + label + "SAR=" + (fmt(event["trigger_sar"]) if event else "NA")])
    fields.extend(["8H=%d" % snapshot["8h"]["bull"],
                   "12H=%d" % snapshot["12h"]["bull"], "D=%d" % d["bull"],
                   "W=%d" % snapshot["1w"]["bull"],
                   "score=" + ("NA" if quality is None else str(quality)),
                   "quality=" + band, ",".join(events) or "WAIT"])
    return "|".join(fields)


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
                if tf in ("8h", "12h", "1d"):
                    live = psar(raw[["open", "high", "low", "close", "volume"]])
                    snapshots[tf]["live_bull"] = bool(live.bull.iloc[-1])
                    snapshots[tf]["live_sar"] = float(live.psar.iloc[-1])
                    snapshots.setdefault("bear_crosses", {})[tf] = bearish_crosses(
                        symbol, raw, now_ms)
                if tf == "1d":
                    snapshots["price"] = float(raw.close.iloc[-1])
                    snapshots["daily_high"] = float(raw.high.iloc[-1])
                    snapshots["daily_live_bull"] = snapshots[tf]["live_bull"]
                    snapshots["daily_live_sar"] = snapshots[tf]["live_sar"]
                    snapshots["daily_cross"] = first_daily_cross(
                        symbol, raw, snapshots[tf], now_ms)
            snapshots["quality_score"] = None
            near_cross = abs(snapshots["price"] / snapshots["1d"]["sar"] - 1) <= .05
            prior = data["symbols"].get(symbol)
            cross = snapshots["daily_cross"]
            new_cross = cross is not None and (
                prior is None or prior.get("last_cross_day") != cross["day"])
            if SCORE_PATH.exists() and (near_cross or new_cross):
                try:
                    history = hourly_history(SESSION, API, symbol, now_ms)
                    score_price = snapshots["price"]
                    if new_cross:
                        event = snapshots["daily_cross"]
                        event_at = pd.Timestamp(event["at"])
                        history = history[history.index + pd.Timedelta(hours=1) <= event_at]
                        score_price = event["price"]
                    snapshots["quality_score"] = entry_score(
                        SCORE_PATH, history, score_price)
                except Exception as score_exc:
                    print("SCOREFAIL|%s|%s" % (symbol, str(score_exc)[:160]), flush=True)
            events = update_symbol(data, symbol, snapshots, timestamp)
            save_state(data)
            print(live_log_line(symbol, snapshots, events), flush=True)
        except Exception as exc:
            failures[symbol] = str(exc)[:160]
            print("LIVEFAIL|%s|%s" % (symbol, failures[symbol]), flush=True)
    print("SCAN|DONE|scanned=%d|failed=%d" % (len(symbols) - len(failures), len(failures)),
          flush=True)


def next_scan_slot(now, last_slot=None):
    """Three local starts daily; ZoneInfo handles Rome's daylight saving time."""
    local = now.astimezone(ROME)
    for offset in (0, 1):
        day = local.date() + timedelta(days=offset)
        for hour in SCAN_HOURS:
            slot = datetime(day.year, day.month, day.day, hour, tzinfo=ROME)
            if slot == last_slot:
                continue
            # A restarted process may catch a slot up to two minutes late.
            if slot >= local or (offset == 0 and 0 <= (local-slot).total_seconds() < 120):
                return slot
    raise AssertionError("no next scan slot")


def main():
    data = load_state()
    print("LIVE|START|state=%s|schedule=10:00,13:00,17:00 Europe/Rome" % STATE,
          flush=True)
    last_slot = None
    while True:
        slot = next_scan_slot(datetime.now(ROME), last_slot)
        print("SCAN|NEXT|%s" % slot.isoformat(), flush=True)
        while True:
            remaining = (slot-datetime.now(ROME)).total_seconds()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 60))
        last_slot = slot
        try:
            scan(data, exchange_universe())
        except Exception as exc:
            print("SCANFAIL|%s" % exc, flush=True)


if __name__ == "__main__":
    main()
