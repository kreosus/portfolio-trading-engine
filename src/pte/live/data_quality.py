"""Data Quality Engine (spec §4). If required data is stale or invalid: NO NEW TRADE, log the reason."""

from __future__ import annotations

import pandas as pd


def check_candles(df: pd.DataFrame, tf_seconds: int, now: pd.Timestamp, cfg: dict, name: str) -> list[str]:
    problems = []
    if df is None or df.empty:
        return [f"{name}: no candles"]
    idx = df.index
    if idx.has_duplicates:
        problems.append(f"{name}: {int(idx.duplicated().sum())} duplicate candles")
    if not idx.is_monotonic_increasing:
        problems.append(f"{name}: timestamps out of order")
    bad = (df.high < df[["open", "close"]].max(axis=1)) | (df.low > df[["open", "close"]].min(axis=1)) | \
          (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        problems.append(f"{name}: {int(bad.sum())} candles with invalid OHLC")
    step = pd.Timedelta(seconds=tf_seconds)
    recent = idx[-200:]
    gaps = pd.Series(recent).diff().dropna() / step - 1
    missing = int(gaps.clip(lower=0).sum())
    if missing > cfg["data"]["max_gap_bars"]:
        problems.append(f"{name}: {missing} missing candles in the last 200")
    age = (now - (idx[-1] + step)).total_seconds()
    if age > tf_seconds + cfg["data"]["stale_seconds"]:        # the bar that just closed must be present
        problems.append(f"{name}: last closed bar ended {age:.0f}s ago (stale)")
    return problems


def check_quote(q: dict, now: pd.Timestamp, cfg: dict) -> tuple[list[str], float]:
    problems = []
    if not q or q.get("bid") is None or q.get("ask") is None:
        return ["quote: missing bid/ask"], float("nan")
    bid, ask = q["bid"], q["ask"]
    if bid <= 0 or ask <= 0 or ask < bid:
        problems.append(f"quote: invalid bid/ask {bid}/{ask}")
    spread_bps = (ask - bid) / ((ask + bid) / 2) * 1e4
    if spread_bps > cfg["execution"]["max_spread_bps"]:
        problems.append(f"quote: spread {spread_bps:.1f} bps > {cfg['execution']['max_spread_bps']} bps")
    age = (now - q["time"]).total_seconds()
    if age > cfg["data"]["stale_seconds"]:
        problems.append(f"quote: {age:.0f}s old (stale)")
    return problems, spread_bps
