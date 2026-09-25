"""Glue: features -> signals -> backtest for a given config."""

from __future__ import annotations

import pandas as pd

from .backtest.costs import CostModel
from .backtest.engine import BacktestResult, run_backtest
from .features.indicators import elevated_vol
from .features.smc import build_features
from .strategies.smc_m15 import generate_signals


def prepare(bars: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict, dict]:
    feats, intents = {}, {}
    for sym, df in bars.items():
        f = build_features(df, cfg["smc"])
        f["elevated_vol"] = elevated_vol(f["atr"], cfg["risk"]["vol_lookback_bars"], cfg["risk"]["vol_percentile"])
        feats[sym] = f
        intents[sym] = generate_signals(f, cfg["strategy"])
    return feats, intents


def backtest(bars: dict[str, pd.DataFrame], funding: dict[str, pd.DataFrame], cfg: dict,
             start=None, end=None, prepared: tuple | None = None) -> BacktestResult:
    feats, intents = prepared if prepared is not None else prepare(bars, cfg)
    return run_backtest(feats, intents, funding, cfg["risk"], CostModel.from_config(cfg["costs"]), start, end)
