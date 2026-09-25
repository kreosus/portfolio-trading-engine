"""Synthetic 15m OHLCV and funding, for tests and pipeline smoke runs only.

Synthetic results say nothing about real edge. A random walk should produce
roughly zero expectancy before costs — a useful sanity check on the backtester.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(n: int = 20_000, seed: int = 0, start: str = "2021-01-01",
              price: float = 30_000.0, vol: float = 0.003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC", name="open_time")
    # regime-switching volatility so ATR percentiles vary
    regime = np.repeat(rng.choice([0.6, 1.0, 1.8], size=n // 500 + 1), 500)[:n]
    steps = 5
    r = rng.normal(0, vol * regime[:, None] / np.sqrt(steps), size=(n, steps))
    path = price * np.exp(np.cumsum(r.ravel())).reshape(n, steps)
    close = path[:, -1]
    open_ = np.concatenate([[price], close[:-1]])
    high = np.maximum(path.max(axis=1), open_)
    low = np.minimum(path.min(axis=1), open_)
    vol_ = rng.lognormal(3, 0.5, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol_, "quote_volume": vol_ * close, "count": 100}, index=idx)


def make_funding(bars: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    times = bars.index[(bars.index.hour % 8 == 0) & (bars.index.minute == 0)]
    return pd.DataFrame({"rate": rng.normal(0.0001, 0.0001, len(times))},
                        index=pd.DatetimeIndex(times, name="time"))
