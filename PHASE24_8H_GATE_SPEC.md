# Phase 24 — ML gate on first 8H bearish PSAR event

Fixed ENTRY: confirmed price cross above Daily PSAR.
Baseline SELL: first 8H PSAR bearish event (Phase23 winner among 67 fixed causal policies).

Question: at an 8H bearish flip, should we SELL now or IGNORE/HOLD until the next eligible exit event?

Optimize realized out-of-sample P&L directly, not classification accuracy.

At each causal 8H bearish flip record only information known at that timestamp: trade return, MFE, giveback, trade age, PSAR states/distances 2H/4H/6H/8H/12H/Daily, bearish breadth, momentum 6/12/24/72H, realized volatility, ATR, ADX/+DI/-DI, EMA slopes/distances, relative volume, swing structure, and BTC context when available.

For each candidate event compute training-only counterfactual rewards: SELL now versus HOLD to next 8H bearish episode / 12H bearish / Daily bearish / terminal horizon. These future rewards are labels only, never features.

Train ExtraTrees and HGB reward-difference models. At test time choose HOLD only when predicted incremental value of HOLD exceeds a threshold calibrated on prior data. Thresholds include a conservative positive margin to avoid ignoring 8H exits without evidence.

Nested expanding chronological walk-forward. Compare every fold against unconditional first-8H SELL on the exact same trades.

Report: geometric/log return, median/mean trade return, p10, giveback, capture, fraction of first-8H flips ignored, and incremental P&L versus baseline.

Promotion rule: ML gate must improve realized OOS log growth over first-8H SELL with acceptable p10/giveback and must not rely on one fold. Otherwise keep simple 8H SELL.
