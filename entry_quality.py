"""Live Phase31 feature extraction and historical percentile scoring."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

FEATURES = (["wb", "wdist", "adx", "didiff", "atr", "e20d", "e50d", "e100d"]
            + ["mom%d" % n for n in (1, 3, 7, 14, 30, 60, 90)]
            + ["vol7", "vol30", "vol90"]
            + [name for tf in (4, 6, 8, 12, 24, 168)
               for name in ("bull%d" % tf, "psard%d" % tf)]
            + ["breadth"])


def research_psar(d, step=.02, maxaf=.2):
    """Phase18C PSAR used to produce the Phase30/31 training features."""
    high = d.high.to_numpy(float)
    low = d.low.to_numpy(float)
    n = len(d)
    bull = np.ones(n, dtype=bool)
    sar = np.full(n, np.nan)
    if not n:
        return bull, sar
    ep, af, sar[0] = high[0], step, low[0]
    for i in range(1, n):
        s = sar[i-1] + af*(ep-sar[i-1])
        if bull[i-1]:
            s = min(s, low[i-1], low[i-2] if i > 1 else low[i-1])
            if low[i] < s:
                bull[i], s, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(maxaf, af+step)
        else:
            s = max(s, high[i-1], high[i-2] if i > 1 else high[i-1])
            if high[i] > s:
                bull[i], s, ep, af = True, ep, high[i], step
            else:
                bull[i] = False
                if low[i] < ep:
                    ep, af = low[i], min(maxaf, af+step)
        sar[i] = s
    return bull, sar


def hourly_history(session, base, symbol, now_ms, days=220):
    start = now_ms - days * 86400000
    rows = []
    while start < now_ms:
        response = session.get(f"{base}/klines", params={
            "symbol": symbol, "interval": "1h", "startTime": start,
            "endTime": now_ms - 1, "limit": 1000}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        start = int(batch[-1][6]) + 1
        if len(batch) < 1000:
            break
    if not rows:
        raise ValueError("no hourly history for score")
    d = pd.DataFrame(rows, columns=[
        "ot", "open", "high", "low", "close", "volume", "ct",
        "q", "n", "tb", "tq", "i"])
    for key in ("open", "high", "low", "close", "volume", "ct"):
        d[key] = pd.to_numeric(d[key])
    d.index = pd.to_datetime(d.ot, unit="ms", utc=True)
    return d[d.ct < now_ms][["open", "high", "low", "close", "volume"]]


def bars(hourly, hours):
    return hourly.resample("%dh" % hours, origin="epoch", label="right",
                           closed="right").agg({
                               "open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()


def adx(d, n=14):
    h, l, c = d.high, d.low, d.close
    up, down = h.diff(), -l.diff()
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    avg = tr.ewm(alpha=1/n, adjust=False).mean()
    plus = up.where((up > down) & (up > 0), 0).ewm(alpha=1/n, adjust=False).mean()
    minus = down.where((down > up) & (down > 0), 0).ewm(alpha=1/n, adjust=False).mean()
    p = 100*plus/avg.replace(0, np.nan)
    m = 100*minus/avg.replace(0, np.nan)
    dx = 100*(p-m).abs()/(p+m).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), p, m, avg


def live_features(history, entry):
    if len(history) < 24*100:
        raise ValueError("less than 100 days of hourly history for Phase31 score")
    weekly = bars(history, 168)
    bw, sw = research_psar(weekly)
    wb = int(bool(bw[-1]))
    wd = (entry-float(sw[-1]))/entry
    daily = bars(history, 24)
    x, p, m, atr = adx(daily)
    c = daily.close
    value = lambda s: float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else 0.
    features = [wb, wd, value(x), value(p)-value(m), value(atr)/entry]
    for span in (20, 50, 100):
        features.append((entry-value(c.ewm(span=span, adjust=False).mean()))/entry)
    for lag in (1, 3, 7, 14, 30, 60, 90):
        features.append(float(entry/c.iloc[-1-lag]-1))
    returns = c.pct_change()
    features.extend(float(returns.tail(n).std()) for n in (7, 30, 90))
    breadth = 0
    for tf in (4, 6, 8, 12, 24, 168):
        bstate, sstate = research_psar(bars(history, tf))
        bull = float(bool(bstate[-1]))
        breadth += bull
        sar = float(sstate[-1])
        features.extend((bull, (entry-sar)/entry if np.isfinite(sar) else 0.))
    features.append(breadth)
    return pd.DataFrame([features], columns=FEATURES).replace([np.inf, -np.inf], 0).fillna(0)


def score(path, history, entry):
    artifact = joblib.load(Path(path))
    if artifact["features"] != FEATURES:
        raise RuntimeError("Phase31 model feature schema mismatch")
    pred = artifact["model"].predict(live_features(history, entry))[0]
    return round(100 * np.searchsorted(artifact["reference"], pred, side="right")
                 / len(artifact["reference"]))
