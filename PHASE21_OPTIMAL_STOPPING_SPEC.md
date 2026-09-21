# Phase 21 — ML Optimal Stopping / Exit Ranking

Fixed premise: ENTRY remains the validated Phase17C daily price > daily PSAR event. Phase21 learns only EXIT.

## Why
Phase19 direct SELL classification generalized moderately (AUC ~0.72) but exited too early. Phase20 absolute future-upside/downside regression collapsed to ~1% exit coverage. The exit problem is therefore reframed as within-trade ranking / optimal stopping rather than point classification or absolute-value regression.

## Causal state vector (sample every 4h)
Only information known at decision time:
- return since entry, running MFE, drawdown from running high, age
- returns/momentum 4/8/12/24/48/72/168h
- realized volatility 12/24/48/72h and volatility acceleration
- distance to running high; time since running high
- PSAR bull/bear state for 1/2/4/6/8/12/24h
- normalized price-to-PSAR distance per TF
- hours since last bearish and bullish flip per TF
- bearish flip count since entry per TF
- transition flags: new bearish flip in last 4/8/12/24h
- number of bearish TFs and weighted bearish breadth
- ordered/cascade summaries across 1→2→4→6→8→12→24h
- re-flip/recovery flags

## Training target
For each historical trade, construct an ex-post exit utility for every candidate decision bar. Future information is used ONLY to create the training target, never as an input feature.

Primary utility should reward realized BUY→exit return and penalize:
1. giveback from the trade's eventual cycle high,
2. material upside left after exit (>5%, >10%, >15%),
3. exits before a substantial profitable impulse has developed.

Convert utility to within-trade percentile/rank. Top-zone labels (e.g. top 10/20/30% bars) may be used as auxiliary targets, but model selection must be based on simulated first causal exit, not row-level AUC alone.

## Validation
Do not reuse a single final holdout repeatedly. Use expanding-window walk-forward folds over the 1,081 Phase17C OOS trades. Each fold trains only on earlier trades and validates on later trades. Aggregate performance across folds and across assets.

Compare at least:
- HistGradientBoosting ranking surrogate
- ExtraTrees ranking surrogate
- pairwise preference model if computationally feasible

At inference, score each new 4h state. Exit only when the score crosses a threshold selected on earlier validation AND persistence/confirmation constraints selected in validation. Threshold must target useful coverage, not permit a degenerate ~1% SELL rate.

## Required reporting
For each fold/model:
- coverage
- median BUY→SELL return
- median captured fraction of available cycle profit
- median giveback from eventual top
- missed-upside P(>5%), P(>10%), P(>15%)
- timing vs daily bearish exit
- per-asset stability
- feature importance / permutation importance

A candidate is promotable only if performance is stable across chronological folds and not dependent on one asset or a tiny subset of trades.
