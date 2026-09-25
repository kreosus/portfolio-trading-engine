# Direction-filter study (exploratory, 2020-2025)

Run 2026-09-25, after the holdout was used. 2020-2025 has been tested many times, so treat this as
idea generation, not evidence. Reproduce with `pte direction --sub-bot <name> --timeframe <tf>`.

**Hypothesis (stated before running):** the candidates leaned on bull-market longs; trading only
with the 200-day trend should keep the edge and lose less when the market turns.

**Modes:** `none` (as registered) · `trend` (longs only above the 200-day average, shorts only
below) · `long_only` (longs above the average, never short). Random entries get the same filter.

| Sub-bot | Mode | Trades | Return | Sharpe | PF | Avg R | Random avg R | Random >= real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trend_pullback 4h | none | 394 | +24% | 0.73 | 1.37 | +0.31 | +0.12 | 3% |
| trend_pullback 4h | trend | 368 | +29% | 0.81 | 1.52 | +0.28 | +0.15 | 3% |
| trend_pullback 4h | long_only | 238 | +29% | 0.84 | 1.72 | +0.41 | +0.29 | 17% |
| funding_momentum 1h | none | 559 | +38% | 0.74 | 1.29 | +0.27 | +0.08 | 0% |
| funding_momentum 1h | trend | 223 | +13% | 0.50 | 1.27 | +0.26 | +0.13 | 0% |
| funding_momentum 1h | long_only | 221 | +14% | 0.54 | 1.30 | +0.27 | +0.13 | 0% |
| vol_breakout 4h | none | 99 | +34% | 0.95 | 2.68 | +0.82 | +0.18 | 0% |
| vol_breakout 4h | trend | 58 | +16% | 0.71 | 2.10 | +0.57 | +0.11 | 10% |
| vol_breakout 4h | long_only | 44 | +9% | 0.48 | 1.75 | +0.45 | +0.35 | 43% |

## Findings

1. **Mostly not supported.** The filter only helps trend_pullback, and modestly. It hurts
   funding_momentum and vol_breakout.
2. **Long-only looks good but is mostly drift.** Random long-only entries earn most of it.
3. **2022 contradicts the simple story.** The unfiltered versions were profitable in the 2022 bear
   market, so their 2025-26 holdout failure is not just a matter of being long in a down market.

## Action

`trend_pullback` 4h with the `trend` filter was added to paper trading as a new entry scored from
2026-09-26 (`paper/PAPER.json`). It is the best of 6 variants on over-used data, so only forward
data can judge it.
