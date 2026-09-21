# Phase 18A — Multi-timeframe SAR sequence around cycle TOP

Baseline: validated Phase 17C SAR cycles.

## Objective
Test whether lower-timeframe PSAR deterioration gives a repeatable early warning of the cycle TOP and anticipates the Daily SAR SELL.

## Locked design
- Reuse the validated Phase 17C cycle universe; do not redefine entries.
- Study window: TOP - 72h through TOP + 72h.
- Timeframes: 1H, 2H, 4H, 6H, 8H, 12H.
- SAR-only in 18A: no MACD, ADX, RSI, volume, news or other filters.
- Reconstruct PSAR causally from information available at each bar.
- TOP is ex-post measurement anchor only, never an input to a live rule.

## Per-cycle/timeframe outputs
1. first bullish-to-bearish PSAR flip in TOP +/-72h;
2. hours from flip to TOP;
3. hours from flip to Daily SELL;
4. new cycle high after flip;
5. maximum additional upside after flip;
6. drawdown after flip before Daily SELL;
7. bullish re-flip before Daily SELL.

## Aggregate outputs
- coverage by timeframe;
- timing distribution vs TOP and Daily SELL;
- false-early and re-flip rates;
- ordered multi-timeframe flip sequences and their frequencies;
- stratification by cycle MFE: <10%, 10-20%, 20-30%, 30-50%, >=50%.

## Leakage rule
Any candidate rule derived from this descriptive study must use only information known at signal time and must subsequently be validated chronologically out of sample.

## Phase 18B gate
Do not optimize a combined exit threshold in 18A. First establish whether a stable SAR sequence exists and quantify the cost of early/false exits.