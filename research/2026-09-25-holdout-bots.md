# Pre-registration: one-shot holdout test of three sub-bots

Registered 2026-09-25, before any holdout data was used. Code at commit `fe2d361`.
Machine-readable version: `2026-09-25-holdout-bots.json` (the runner reads it; nothing is typed at run time).

## What is tested

| Sub-bot | Timeframe | Parameters | Config hash |
| --- | --- | --- | --- |
| trend_pullback | 4h | stop 2.0 ATR, trail 3.0 ATR, time exit 480 bars | ae3192e9a0d7 |
| funding_momentum | 1h | funding baseline 0.01%, stop 3.0 ATR, time exit 96 bars | ed8b74799c20 |
| vol_breakout | 4h | squeeze 20th pct, volume 1.5x, stop 1.5 ATR, trail 2.5 ATR, time exit 192 bars | ae3192e9a0d7 |

- Window: 2025-09-01 to 2026-08-31 (the 12 locked months, never used before).
- BTCUSDT and ETHUSDT perpetuals, $10,000 each, trading independently.
- Live risk rules (8% drawdown halt and loss-streak pause ON). Costs as in `config/default.yaml`.
- Why these three: they were the best of 15 sub-bot/timeframe combinations tested on 2020-2025.
  That selection itself is a reason to expect some decay out of sample.

## Pass/fail bar (all must hold)

| Criterion | trend_pullback 4h | funding_momentum 1h | vol_breakout 4h |
| --- | --- | --- | --- |
| Trades (half of 2020-2025 annual rate) | >= 36 | >= 52 | >= 9 |
| Net return after costs | > 0 | > 0 | > 0 |
| Profit factor | >= 1.10 | >= 1.10 | >= 1.10 |
| Profit factor at 2x costs | >= 1.00 | >= 1.00 | >= 1.00 |
| 8% drawdown halt not triggered | yes | yes | yes |

Reported but not scored: Sharpe (too noisy over 12 months), max drawdown, win rate, average R,
and the same run with the halt off.

## What the outcome means

- **Pass:** candidate for Phase 5 paper trading, after the 2020-2025 robustness checks.
  A vol_breakout pass (about 18 trades a year) means it survived, with low confidence.
- **Fail:** shelved. No re-tuning against the holdout.
- The holdout is used once. After this run it is no longer out-of-sample for these rules.
