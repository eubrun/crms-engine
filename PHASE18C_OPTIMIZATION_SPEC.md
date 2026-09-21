# Phase 18C — Causal combinatorial SAR exit optimization

Goal: discover, rather than hand-pick, the SAR state/sequence rule that best preserves cycle profit on the exact 1,081 Phase17C OOS events.

## Candidate feature space
At every causal timestamp after BUY, encode for TF = 1H,2H,4H,6H,8H,12H:
- current PSAR state (bull/bear)
- bearish flip count since BUY (1st, 2nd, 3rd+)
- hours since latest bearish flip
- persistence in bearish state
- whether a bullish re-flip occurred since previous bearish flip
- number of simultaneously bearish TFs
- ordered recent bearish-flip sequence

Explicitly include combinations such as:
- 1st/2nd/3rd 4H bearish flip
- 4H bearish -> 12H bearish
- 4H bearish + 6H bearish
- 2nd 4H bearish + 6H/8H/12H confirmation
- simultaneous bearish sets (4+6, 4+8, 4+12, 4+6+8, 4+6+8+12)
- ordered cascades across 1/2/4/6/8/12H.

## Objective
For each candidate exit rule compute realized return from BUY to exit and ex-post opportunity cost from exit to cycle TOP. Primary loss thresholds: remaining upside >5%, >7.5%, >10%, >15%. Also record coverage, median return, median giveback from TOP, timing vs Daily SELL, and false-exit/recovery behavior.

Optimization must not simply maximize in-sample return. Use chronological nested validation inside the 1,081 OOS events: earlier portion selects candidate/parameters; later untouched portion evaluates it. Penalize low coverage and rule complexity. Report Pareto frontier (profit capture vs missed-upside risk vs coverage), not only one winner.

No TOP-derived feature may be used to trigger an exit. TOP is evaluation-only.
