# Phase 21B — Trend-strength false-exit reduction

Baseline: Phase21 optimal stopping. Fixed ENTRY = daily price > daily PSAR. Do not redesign entry.

Goal: reduce premature exits that are followed by >10% / >15% additional upside, while preserving useful exit coverage.

Add causal features at each 4h decision point:
- ADX, +DI, -DI on 4H/8H/12H/24H
- ADX slope / acceleration and DI spread (+DI - -DI)
- EMA 20/50 slopes and normalized price distances on 4H/12H/24H
- rolling high/low structure: higher-high / higher-low counts over 24/48/72h
- relative volume vs 24/72h baseline and volume acceleration
- momentum acceleration: delta of 6/12/24/72h momentum
- ATR-normalized distance from running high
- persistence of bearish SAR breadth and whether breadth is expanding or contracting
- time since running high and number of new highs in last 24/48h

Training remains within-trade exit-quality ranking with expanding chronological walk-forward. Add auxiliary false-exit target = future upside >10% after candidate exit, used only from training history. Candidate score combines predicted exit quality with probability of false exit.

Model selection constraints:
- calibration exit coverage >= 50%
- prioritize reducing P(missed upside >10%) and >15%
- reject if median BUY→SELL return materially degrades versus Phase21 baseline
- report every fold and aggregate, plus per-asset stability if candidate improves aggregate.

Phase21 baselines:
ET: cov .542, ret .0570, miss .0826, give .0912, miss10 .458, miss15 .354
HGB: cov .502, ret .0594, miss .0759, give .0784, miss10 .443, miss15 .345
