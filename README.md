# CRMS Engine

Quantitative crypto research, backtesting and daily signal engine.

Initial objectives:
- ingest closed daily OHLCV candles from Binance-compatible sources;
- compute Parabolic SAR daily/weekly, MACD/histogram, RSI, ADX/+DI/-DI, moving averages, ATR and RVOL;
- measure relative strength vs BTC and cross-asset strength;
- backtest PSAR flips and multi-factor confirmations without look-ahead bias;
- report forward returns (5/10/20/30/60 bars), MAE, MFE, retests and failure rates;
- support walk-forward/out-of-sample validation;
- scan a liquid crypto universe for DISCOVERY, PRE-SIGNAL, TRIGGER, EXPANSION and EXHAUSTION states;
- emit machine-readable JSON/CSV suitable for scheduled Railway execution and ChatGPT analysis.

The project is research/paper-trading infrastructure first. Live execution is intentionally out of scope for the initial version.

## Live scanner branch

`sar_live_scanner.py` discovers currently tradable Binance spot USDT pairs
through `exchangeInfo` at the start of each scan. Known fiat, stablecoin and
tokenized equity bases are excluded; the remaining pairs form the crypto
universe. The priority list is scanned first and reported
alongside missing Binance listings; it does not restrict the scan. Scans start
at 10:00, 13:00 and 17:00 Europe/Rome daily, including daylight saving
changes. The first
observation of a pair can reconstruct an intraday cross from today's one-minute
candles: the day's high must exceed the last closed bearish Daily SAR, and the
first crossing minute sets a theoretical entry at the higher of its open and
that SAR. This captures a touch followed by a retreat before scan time. The
log shows both the frozen trigger SAR and the current candle's provisional
Daily SAR state/value. A late discovery is a historical paper signal, not a
claim that the crossing price remains executable. Existing paper trades follow the
frozen Weekly-at-entry, 8H/12H/Daily exit hierarchy. Bearish exits use the
low of each 8H, 12H or Daily candle against its preceding closed bullish SAR,
including a crossing followed by a recovery before the scan. The first crossing
minute and theoretical crossing price are recorded as `exited_at` and `exit`;
`exit_detected_at` is the later scan time. Logs expose `bear8`, `bear12` and
`bearD`, plus `bear8Px/bear8SAR` (and corresponding 12/D fields). `crossPx`
and `crossSAR` describe the Daily buy event. `px` is the price observed at
the scan, while `sarLive8`, `sarLive12` and `sarLiveD` are recalculated with
the still-open candle; these new SAR values and states are provisional until
that candle closes. Paper fills reconstructed after the fact are not executable
prices.

Set `CRMS_STATE_PATH` to a path on a persistent Railway volume before running
the scanner. The default `output/live_state.json` is suitable for local runs
but will not survive a Railway replacement without a volume. Keep only one
scanner instance writing that file. The process prints `LIVEFAIL` for pairs
that cannot be scanned and reports coverage at the end of each pass.

The Phase 31 Entry Quality Score is a trained ExtraTrees percentile, rather
than a fixed weighted checklist. Run `python -u train_live_score.py` on
historical data to save the fitted model and its reference predictions at
`CRMS_SCORE_PATH` (default `/data/phase31_score.joblib`). Training is an
explicit, resource-intensive job. It must finish successfully before a score
is published. At each new cross and for candidates within 5% of the Daily
SAR, the scanner loads that artifact and computes the 31 Phase30 features.
If the artifact is absent or scoring fails, it reports `score=NA`, while BUY
remains unconditional. The Docker default still runs historical research, so
a Railway service must explicitly launch `python -u sar_live_scanner.py`.
