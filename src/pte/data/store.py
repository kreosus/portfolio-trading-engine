"""Load processed data for research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_bars(symbol: str, processed_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(processed_dir) / f"{symbol}_15m.parquet")


def load_funding(symbol: str, processed_dir: str | Path) -> pd.DataFrame:
    p = Path(processed_dir) / f"{symbol}_funding.parquet"
    if not p.exists():
        return pd.DataFrame({"rate": []}, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    return pd.read_parquet(p)


def split_holdout(df: pd.DataFrame, holdout_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the most recent `holdout_months` whole months as a locked holdout."""
    if df.empty or holdout_months <= 0:
        return df, df.iloc[0:0]
    last_month_start = df.index[-1].tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    cut = last_month_start - pd.DateOffset(months=holdout_months - 1)
    return df[df.index < cut], df[df.index >= cut]


TIMEFRAMES = {"15m": "15min", "1h": "1h", "4h": "4h"}
HTF_FOR = {"15m": "1h", "1h": "4h", "4h": "1D"}   # SMC trend filter: one step up


def resample_bars(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate 15m bars to a higher timeframe, keeping only COMPLETE bars."""
    rule = TIMEFRAMES[timeframe]
    if rule == "15min":
        return df
    agg = df.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum", "count": "sum"})
    n = df["close"].resample(rule, label="left", closed="left").count()
    need = int(pd.Timedelta(rule) / pd.Timedelta("15min"))
    return agg[n == need].dropna()
