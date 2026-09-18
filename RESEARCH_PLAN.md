# SAR Research Program

Goal: identify a sparse, robust subset of Daily Parabolic SAR bullish flips with high precision, while preserving the exit rule: exit on the next bearish Daily SAR flip.

## Non-negotiable validation
- Strict chronological train / validation / untouched OOS test.
- Walk-forward validation after discovery.
- Minimum sample counts and confidence intervals; never accept 70-80% from tiny samples.
- Include transaction costs/slippage and 1-bar execution-lag robustness.
- Parameter perturbation tests around every winning threshold.
- Report precision (win rate), mean/median return, payoff ratio, expectancy, drawdown, trade count/year, duration, MAE/MFE.
- Prefer stable plateaus over a single optimized parameter point.

## Research families
1. SAR geometry: AF state, distance price-SAR normalized by ATR, prior bearish-run age, flip gap, SAR acceleration and curvature.
2. Trend state: ADX level/slope/acceleration, +DI/-DI spread and slopes, EMA slopes/stacks, Donchian position, Aroon, Supertrend state.
3. Momentum state: RSI level/slope/divergence, MACD histogram slope/zero-cross distance, ROC across 3/5/10/20/40/60d, stochastic, CCI.
4. Volatility state: ATR percentile, Bollinger bandwidth percentile/squeeze-release, realized volatility, upside/downside semivariance, range compression/expansion.
5. Volume/liquidity: RVOL, OBV slope, CMF/MFI, volume-price divergence, turnover proxies where available.
6. Market regime: BTC trend/momentum/volatility/drawdown; ETH confirmation; breadth across universe; alt/BTC relative strength; cross-sectional rank.
7. Multi-timeframe: weekly SAR/trend; slower 20/40/60/84d momentum as regime filters; disagreement states between fast and slow signals.
8. Asset specialization: per-asset and clustered asset-family models; exclude historically hostile SAR assets if OOS evidence supports it.
9. Meta-labeling: classify each SAR flip as take/skip using only features known at entry; compare interpretable rules with regularized logistic regression, shallow trees, random forest/gradient boosting with strict walk-forward OOS.
10. Trade-path labels: winner at SAR exit plus MAE/MFE, time-to-profit, target-before-stop variants; do not alter canonical SAR exit benchmark unless separately labelled.

## Promotion gates
Candidate A: OOS win rate >= 80% with adequate sample and stable walk-forward results.
Candidate B: if A unavailable, OOS win rate >= 70% with strong positive expectancy, acceptable drawdown and adequate frequency.
Anything below 70% remains research-only regardless of in-sample performance.

## Iteration order
A. Build richer event dataset for every bullish SAR flip.
B. Run univariate diagnostics and asset/regime decomposition.
C. Search interpretable 2-6 condition rules.
D. Train meta-label classifiers with nested/walk-forward tuning.
E. Stress test winners with costs, lag, threshold perturbation and regime slices.
F. Freeze final rule before evaluating the last untouched holdout.
