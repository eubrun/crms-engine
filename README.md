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
