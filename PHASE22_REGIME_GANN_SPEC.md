# Phase 22 — Regime-Aware + Gann Exit

Baseline remains Phase21 HGB. Entry remains fixed: price crosses above Daily PSAR.

## Objective
Determine whether explicit market-regime conditioning plus causal Gann timing/price geometry improves exit timing, especially false exits followed by >10% / >15% additional upside.

## Per-asset Gann clocks
Compute separately for every cryptocurrency using only information available up to each decision timestamp.

Identify confirmed swing highs/lows on 4H, 12H, Daily and Weekly-like aggregates using volatility-normalized pivot logic. Never use a future bar to identify a pivot at the current timestamp; pivot confirmation delay is part of the feature.

For each decision state measure time since:
- Phase17C BUY
- last confirmed significant swing low
- last confirmed significant swing high
- prior major cycle high/low

Candidate Gann time cycles: 7, 14, 21, 30, 45, 60, 90 calendar days. Encode distance to nearest cycle/window rather than hard-coded SELL rules. Also encode harmonic multiples / nearest-cycle identity.

## Gann price geometry
- volatility/ATR-normalized 1x1-style slope from confirmed pivots (not chart-pixel 45-degree geometry)
- normalized distance above/below projected pivot slope
- Square-of-9-inspired transformed-price distance features, tested as features only
- price/time confluence score: proximity to both a time-cycle window and a normalized projected price level

## BTC context
For every altcoin decision state include causal BTC features at the same timestamp:
- BTC Daily/12H/4H PSAR state
- BTC trend regime / ADX-momentum state
- BTC time since confirmed major swing high/low
- BTC distance to nearest Gann time window
- BTC normalized Gann price geometry/confluence
For BTC trades these context fields equal BTC's own state.

## Regime layer
Learn/derive causal regime descriptors from volatility, ADX/DI, EMA slope, SAR breadth, momentum, drawdown and BTC context. Test both:
1. one global exit model with regime probabilities/features;
2. regime-conditioned models if sample size per regime is sufficient.
Do not force named clusters to be economically meaningful unless walk-forward evidence supports them.

## Ablation experiment
On identical expanding walk-forward folds compare:
A. Phase21-like baseline features
B. A + regime features
C. A + Gann features
D. A + regime + Gann + BTC context

Promotion requires D (or another augmented set) to improve out-of-sample results consistently, especially miss10/miss15, without collapsing coverage or median BUY→SELL return. Gann is retained only if its incremental walk-forward contribution survives ablation.

Phase21 HGB baseline: coverage .502, median return .0594, residual upside .0759, giveback .0784, miss10 .443, miss15 .345.
