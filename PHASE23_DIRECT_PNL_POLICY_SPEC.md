# Phase 23 — Direct P&L Optimal Exit Policy

## Fixed premise
ENTRY is immutable: confirmed price cross above Daily Parabolic SAR (Phase17C universe, 1,081 OOS trades).

## Single objective
Find the causal EXIT policy that maximizes realized out-of-sample trading performance. Stop optimizing proxy metrics as the primary objective.

Primary model-selection objective on chronological validation:
- compounded realized return / log-growth across trades
Secondary safeguards:
- max drawdown / lower-tail trade return
- stability across chronological folds and assets
- minimum practical exit coverage only where required to avoid undefined policies

## Oracle diagnostic
For every trade calculate ex-post:
- maximum favorable excursion (MFE)
- best feasible 4h close after entry (oracle exit)
- oracle return
- realized/oracle capture ratio for each tested policy
Oracle is diagnostic only and is NEVER an input feature.

## Policy families to compete
1. PSAR exits: 1/2/4/6/8/12/24h, first bearish flip, persistence, second bearish episode, ordered cascades and breadth thresholds.
2. Profit-aware PSAR: same signals conditioned on MFE/realized return/drawdown from running high.
3. Volatility trailing: ATR multiples, Chandelier-style trailing, percentage trailing stops scaled by volatility.
4. Market structure: break of causal swing low / higher-low, EMA20/50 loss and slope, Supertrend-style state.
5. Momentum/trend deterioration: ADX/+DI/-DI, MACD, RSI, momentum acceleration and combinations with PSAR breadth.
6. Time/regime context: age, volatility regime, BTC context and Gann features as optional competitors, not assumptions.
7. ML policy: HOLD/SELL score from causal state vector, threshold/persistence selected only on prior validation data.

## Evaluation protocol
Nested expanding walk-forward. Hyperparameters/policy are selected on historical training/calibration only and then frozen on the next unseen block. Do not repeatedly optimize on the same final holdout.

For every policy/fold report:
- number of trades and exit coverage
- arithmetic median trade return
- compounded/log growth score
- max/lower-tail drawdown proxy
- median capture ratio vs oracle
- median giveback from MFE
- missed upside after exit
- per-asset dispersion

## Mandatory benchmarks
- bearish Daily PSAR exit
- 12H, 8H, 6H, 4H PSAR bearish exits
- best simple SAR persistence/cascade selected only in training
- Phase21 HGB

Promotion criterion: maximize walk-forward realized P&L subject to robustness. A more complex ML policy is rejected if a simpler causal rule produces equal or better OOS P&L with comparable robustness.
