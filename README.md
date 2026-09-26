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

`sar_live_scanner.py` discovers every currently tradable Binance spot USDT pair
through `exchangeInfo` at the start of each scan. The priority list is reported
alongside missing Binance listings; it does not restrict the scan. The first
observation of a pair initializes its baseline; only a subsequent below-to-above
Daily SAR transition generates a paper BUY. Existing paper trades follow the
frozen Weekly-at-entry, 8H/12H/Daily exit hierarchy.

Set `CRMS_STATE_PATH` to a path on a persistent Railway volume before running
the scanner. The default `output/live_state.json` is suitable for local runs
but will not survive a Railway replacement without a volume. Keep only one
scanner instance writing that file. The process prints `LIVEFAIL` for pairs
that cannot be scanned and reports coverage at the end of each pass.

The Phase 31 Entry Quality Score is a trained ExtraTrees percentile, rather
than a fixed weighted checklist. No trained model artifact is checked into
this repository. The scanner reports `score=NA` until causal model training,
feature parity, and artifact deployment are completed; BUY signals remain
unconditional under the frozen rule. The Docker default still runs historical
research, so a Railway service must explicitly launch
`python -u sar_live_scanner.py`.
