# Result: one-shot holdout test of three sub-bots

Pre-registration: `2026-09-25-holdout-bots.md`, pushed as commit `24986af` before the run.
Run once on 2026-09-25 with `pte holdout-bots --confirm`. Raw record: `2026-09-25-holdout-bots-RESULT.json`.

## Verdict: all three FAIL. Shelved.

| Sub-bot | Trades | Return | Profit factor | PF at 2x costs | Halted | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| trend_pullback 4h | 65 | -2.1% | 0.83 | 0.80 | no | FAIL |
| funding_momentum 1h | 69 | -4.8% | 0.65 | 0.63 | 2026-03-08 | FAIL |
| vol_breakout 4h | 24 | -0.0% | 0.99 | 0.92 | no | FAIL |

Reported, not scored (halt off): trend_pullback -0.5% (PF 0.97), funding_momentum -2.7% (PF 0.88),
vol_breakout +3.2% (PF 1.42, 25 trades).

Market over the holdout (Sep 2025 - Aug 2026): BTC -27%, ETH -44%, with lower volatility than 2020-2025.
The holdout is now used. Only future data (paper trading) is out-of-sample from here.

## Diagnosis on 2020-2025 data (after the holdout, research mode)

| | trend_pullback 4h | funding_momentum 1h | vol_breakout 4h |
| --- | --- | --- | --- |
| Settings moved +/-25%: share still profitable | 100% | 100% | 100% |
| Mean R per trade (bootstrap 95% CI) | +0.31 (+0.06..+0.60) | +0.27 (+0.07..+0.47) | +0.82 (+0.17..+1.60) |
| Longs / shorts, mean R | +0.44 / +0.06 | +0.32 / -0.04 | +1.06 / +0.31 |
| Random entries, same exits: median mean R | +0.18 | +0.10 | +0.14 |
| Random runs matching or beating the real signal | 7.5% | 0% | 5% |
| Profitable calendar years | 3 of 6 | 4 of 6 | 6 of 6 |

What this says:

1. **Not a lucky setting.** Every +/-25% variant stayed profitable, so the 2020-2025 results sit on a plateau.
2. **Much of the "edge" was market drift.** Random entries with the same stops and exits earned
   +0.10R to +0.18R per trade in 2020-2025, a strong bull market. The profits lean heavily on longs.
3. **The signals added something on top of that in 2020-2025,** but the bootstrap p-values (about
   0.003-0.009) are not corrected for the roughly 30 configurations tried, and the edge did not survive
   a bear-market year.

Reproduce with `pte robustness --sub-bot <name> --timeframe <tf>`.
