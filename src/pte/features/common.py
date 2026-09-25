"""Shared causal features for the non-SMC strategies."""

from __future__ import annotations

import pandas as pd

from .indicators import atr, ema, rsi, session_vwap


def funding_on_bars(df: pd.DataFrame, funding: pd.DataFrame | None) -> pd.Series:
    """Most recent SETTLED funding rate known at each bar's close (as-of join)."""
    if funding is None or funding.empty:
        return pd.Series(0.0, index=df.index, name="funding_rate")
    close_time = pd.DataFrame({"t": df.index + pd.Timedelta("15min")})
    fr = funding["rate"].rename("funding_rate").reset_index().rename(columns={funding.index.name or "index": "ft"})
    fr.columns = ["ft", "funding_rate"]
    fr["ft"] = fr["ft"].astype("datetime64[ns, UTC]")
    close_time["t"] = close_time["t"].astype("datetime64[ns, UTC]")
    m = pd.merge_asof(close_time, fr.sort_values("ft"), left_on="t", right_on="ft", direction="backward")
    return pd.Series(m["funding_rate"].fillna(0.0).to_numpy(), index=df.index, name="funding_rate")


def build_common(df: pd.DataFrame, funding: pd.DataFrame | None = None, atr_period: int = 14) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["atr"] = atr(df, atr_period)
    c = df["close"]
    for span in (50, 200, 800):
        out[f"ema{span}"] = ema(c, span)
    out["rsi"] = rsi(c, 14)
    out["vwap"] = session_vwap(df)
    out["vol_ma48"] = df["volume"].rolling(48).mean()
    out["hh48"] = df["high"].rolling(48).max().shift(1)      # prior 48 bars, excludes current
    out["ll48"] = df["low"].rolling(48).min().shift(1)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bbw = (4 * sd / mid)
    out["bbw"] = bbw
    out["bbw_pct"] = bbw.rolling(2880, min_periods=720).rank(pct=True)
    out["roc288"] = c / c.shift(288) - 1                        # 3-day return
    out["funding_rate"] = funding_on_bars(df, funding)
    return out
