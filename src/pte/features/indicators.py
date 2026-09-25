"""Causal indicators. Every value at bar i uses bars <= i only."""

from __future__ import annotations

import pandas as pd


def bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Bar length, inferred as the most common spacing of the first bars."""
    if len(index) < 2:
        return pd.Timedelta("15min")
    return pd.Series(index[:1000]).diff().dropna().mode().iloc[0]


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename("atr")


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return (100 - 100 / (1 + up / dn)).rename("rsi")


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP reset at each UTC day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    return (pv / v).rename("vwap")


def elevated_vol(atr_s: pd.Series, lookback: int, pct: float) -> pd.Series:
    """True when ATR is above its trailing `pct` quantile (window includes current bar)."""
    q = atr_s.rolling(lookback, min_periods=lookback // 4).quantile(pct)
    return (atr_s > q).fillna(False).rename("elevated_vol")
