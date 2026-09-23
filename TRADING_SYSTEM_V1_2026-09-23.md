# Trading System Crypto v1 — frozen 2026-09-23

## Entry
- BUY candidate: price crosses above Parabolic SAR Daily.
- Entry Quality Score 0–100 ranks simultaneous candidates; it is not a hard filter.
- Priority: 80–100 high; 60–79 medium; <60 low.

## Exit
State of Weekly SAR is recorded at BUY.
- Weekly bearish at BUY: SELL at first bearish SAR 8H.
- Weekly bullish at BUY: at first bearish SAR 8H calculate P&L from entry:
  - P&L < +4%: SELL at 8H flip.
  - +4% <= P&L < +10%: ignore 8H flip, SELL at bearish SAR 12H.
  - P&L >= +10%: ignore 8H/12H noise, SELL at bearish SAR Daily.

## Evidence
- Phase 27: Weekly regime materially changes optimal exit; 8H strongest when Weekly bearish.
- Phase 28: simple 3%/8% hierarchy beat fixed 8H in 3/4 OOS periods.
- Phase 29: 35 threshold pairs tested (low 2–5%, high 8–12%); 35/35 positive in aggregate, 32/35 won >=3/4 chronological folds. Robust center selected at 4%/10%.
- Phase 30: hard ML entry filtering improved selected-trade quality but risked losing large winners; not promoted.
- Phase 31: walk-forward ExtraTrees percentile score. OOS 80–100 band returned +4.23% geometric/trade with P10 -3.53%. On 150 same-day multi-cross days, top score returned +1.99% geometric/trade vs +0.98% for peers; useful for ranking, not as a veto.

## Live forward test
Start: 2026-09-23. Scan at 10:00, 13:30 and 17:00 Europe/Rome. Track timestamp, entry, Daily/Weekly SAR state, score, max excursion, first 8H flip, P&L at flip, assigned exit timeframe and final exit. Change rules only after new walk-forward evidence; avoid ad-hoc overfitting.

## Priority universe
BCH, LTC, SOL, SUI, ICP, DOT, XRP, AVAX, XLM, ZEC, HYPE, DYDX, ONDO, INJ, RAY, plus scanner-supported assets with sufficient history.
