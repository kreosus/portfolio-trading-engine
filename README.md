# portfolio-trading-engine

Research-first systematic trading engine. **Phase 1** answers one question:

> Does the M15 SMC setup have positive expectancy on BTC and ETH perpetuals after fees, slippage and funding?

Everything here is built to make that answer honest: causal features, conservative fills, realistic costs,
walk-forward validation, a locked holdout, and kill criteria fixed before any result is seen.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                               # 29 tests, including look-ahead checks

pte download                         # Binance USD-M archives -> data/raw (BTCUSDT, ETHUSDT, 15m + funding)
pte process                          # raw zips -> data/processed/*.parquet
pte backtest                         # single run on development data (holdout excluded)
pte walkforward                      # walk-forward + kill-criteria verdict -> reports/
pte holdout --confirm                # opens the locked holdout ONCE, at the end of Phase 1
```

Pipeline check without real data: `pte walkforward --synthetic 110000` (a random walk should fail the kill criteria).

## Layout

```
config/default.yaml          every tunable, fixed before testing
src/pte/data/                Binance archive downloader (checksummed), parquet store, holdout split, synthetic data
src/pte/features/            causal indicators + SMC detectors (swings, BOS/CHoCH, sweeps, displacement, FVG, OB, HTF bias)
src/pte/strategies/smc_m15   Strategy A: sweep -> displacement + BOS/CHoCH -> FVG limit entry
src/pte/risk.py              risk engine: drawdown ladder, daily/weekly limits, open-risk and leverage caps, halt
src/pte/backtest/            event-driven portfolio backtester, cost model, metrics
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
