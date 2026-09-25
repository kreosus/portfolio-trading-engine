# portfolio-trading-engine

Research-first systematic trading engine. **Phase 1** answers one question:

> Does the M15 SMC setup have positive expectancy on the BTCUSDT perpetual after fees, slippage and funding?

**Bitcoin only.** Since 2026-09-25 the system trades BTCUSDT only (`data.symbols` in the config).
Research notes dated before that used BTCUSDT + ETHUSDT and say so.

Everything here is built to make that answer honest: causal features, conservative fills, realistic costs,
walk-forward validation, a locked holdout, and kill criteria fixed before any result is seen.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                               # 47 tests, including look-ahead checks

pte download                         # Binance USD-M archives -> data/raw (BTCUSDT 15m + funding)
pte process                          # raw zips -> data/processed/*.parquet
pte backtest                         # single run on development data (holdout excluded)
pte walkforward                      # walk-forward + kill-criteria verdict -> reports/
pte bots                             # all five sub-bots, each on the full account, reported separately
pte bots --timeframe 4h              # same rules on 1h or 4h bars (params are in bars: 4x / 16x the time)
pte bots --research-mode             # measurement only: drawdown halt and loss-streak pause off
pte holdout --confirm                # opens the locked holdout ONCE, at the end of Phase 1
pte holdout-bots --confirm           # one-shot holdout test of the sub-bots pre-registered in research/
pte robustness --sub-bot vol_breakout --timeframe 4h   # settings plateau, splits, bootstrap, random-entry baseline
pte direction --sub-bot trend_pullback --timeframe 4h  # 200-day direction filter vs none
pte paper update                     # forward paper trading (normally run daily by GitHub Actions)
```

## Paper trading

`paper/PAPER.json` registers which sub-bots are paper-traded forward, from when, and the review bar.
The current registry is Bitcoin only (`paper-btc-2026-09-26`); the earlier BTC + ETH one is in `paper/archive/`.
A GitHub Actions workflow (`.github/workflows/paper.yml`) runs daily: it downloads new archive data
from data.binance.vision, re-runs every entry with the same engine as the backtests, and commits
the results to the **`paper-results`** branch (`README.md` there is the scoreboard).

- Prices: daily 15m archive files, about one day behind real time.
- Funding: estimated from the 1m premium index until Binance publishes the monthly file, so the
  current month is provisional (the estimate matched the real rate's side of 0.01% in 98-100% of
  settlements in Jul-Aug 2026).
- Entries are never edited. New rules get a new entry with its own start date; the runner refuses
  to run if an existing entry's rules change.
- Run it locally with `pte paper update`. No money is involved.

## Sub-bots

One bot, five sub-bots. Each trades the **full account on its own** — own capital, risk engine, positions,
trades and equity — and is reported separately. There is no combined portfolio. Rules are in
`src/pte/strategies/` and parameters under `bots:` in `config/default.yaml` (fixed before testing).

| Sub-bot | Idea |
| --- | --- |
| `smc` | Phase 1 M15 liquidity/SMC (shelved: failed kill criteria) |
| `trend_pullback` | EMA200 > EMA800 trend, EMA50 reclaim, ATR trailing stop |
| `vol_breakout` | Bollinger squeeze, 48-bar breakout on volume, ATR trailing stop |
| `vwap_reversion` | > 2.5 ATR from day VWAP, RSI extreme, exhaustion bar, target VWAP |
| `funding_momentum` | 3-day momentum at funding settlements, filtered by uncrowded funding |

Disable one with `enabled: false`. Add one by writing a function `f(features, params) -> list[OrderIntent]`
in `strategies/library.py`, registering it in `REGISTRY`, and adding its block to the config.

Pipeline check without real data: `pte walkforward --synthetic 110000` (a random walk should fail the kill criteria).

## Layout

```
config/default.yaml          every tunable, fixed before testing
src/pte/data/                Binance archive downloader (checksummed), parquet store, holdout split, synthetic data
src/pte/features/            causal indicators + SMC detectors (swings, BOS/CHoCH, sweeps, displacement, FVG, OB, HTF bias)
src/pte/strategies/smc_m15   Strategy A: sweep -> displacement + BOS/CHoCH -> FVG limit entry
src/pte/risk.py              risk engine: drawdown ladder, daily/weekly limits, open-risk and leverage caps, halt
src/pte/backtest/            event-driven backtester (independent sub-bots), cost model, metrics
src/pte/bots.py              builds and reports the five sub-bots
src/pte/research/            walk-forward, experiment log, kill criteria
```

## Anti-cheating rules (enforced in code/tests)

- Features at bar *i* use bars ≤ *i* only. `tests/test_causality.py` recomputes every feature and signal on
  truncated data and fails if any past value changes.
- Swing points are recorded on their confirmation bar (k+N), not the swing bar.
- HTF bias uses only completed 1h bars.
- Limits fill only when price trades *through*; gaps through a stop fill at the open; stop+target in one bar = stop.
- Fees on every fill, slippage on taker fills, funding at each 8h settlement; OOS also reported at 2× costs.
- The grid (swing N × target R) is chosen on each train fold and scored only on the next test fold.
- Every configuration evaluated is appended to `reports/experiments.jsonl` (multiple-testing record).
- The last 12 months are excluded from all research until `pte holdout --confirm`, which can run once.

## Kill criteria (config `kill_criteria`)

OOS trades ≥ 200 · profit factor > 1.2 · Sharpe > 1.0 · PF at 2× costs > 1.0 · ≥ 60% profitable folds ·
holdout year positive. Fail any → shelve, don't re-tune.

Own-account research only. Not financial advice. Fee defaults reflect Binance USD-M VIP0; verify current rates.
