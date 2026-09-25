"""Strategies B–E. Each is independent, with rules fixed before testing.

All signals fire on a closed bar and enter at the next bar's open (market),
unless noted. Parameters live in config/default.yaml under `bots`.

B trend_pullback   15m EMA200 > EMA800 (≈ 2-day > 8-day trend). Long when the
                   close reclaims EMA50 after the prior close was below it,
                   RSI > 50. Stop 2 ATR, trailing 3 ATR, time exit 5 days.
                   Short mirrors it.
C vol_breakout     Bollinger bandwidth in its lowest 20% of the last 30 days
                   (compression), then a close beyond the prior 48-bar high/low,
                   closing in the outer 25% of its bar, on volume > 1.5x the
                   48-bar average. Stop 1.5 ATR, trailing 2.5 ATR, time exit 2 days.
D vwap_reversion   Close more than 2.5 ATR below the UTC-day VWAP, RSI < 25,
                   and the bar closes up (exhaustion). Target = VWAP at signal,
                   stop 1.5 ATR, time exit 8 hours. Short mirrors it.
E funding_momentum Evaluated only on 8-hour funding settlement bars. Long when
                   the 3-day return > 0 and the last funding rate is below the
                   0.01% baseline (longs not crowded). Short when the 3-day
                   return < 0 and funding is above the baseline. Stop 3 ATR,
                   time exit 1 day.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .smc_m15 import OrderIntent

INF = math.inf


def _mk(i, side, close, stop_dist, time_exit, target=None, trail=None):
    stop = close - side * stop_dist
    tgt = target if target is not None else side * INF
    return OrderIntent(i, side, close, stop, tgt, i + 1, time_exit, entry_type="market", trail_atr=trail)


def trend_pullback(f: pd.DataFrame, p: dict) -> list[OrderIntent]:
    c = f["close"].to_numpy(); e50 = f["ema50"].to_numpy()
    up = (f["ema200"] > f["ema800"]).to_numpy(); dn = (f["ema200"] < f["ema800"]).to_numpy()
    r = f["rsi"].to_numpy(); a = f["atr"].to_numpy()
    out = []
    for i in range(1, len(f)):
        if np.isnan(a[i]) or np.isnan(e50[i]) or np.isnan(f["ema800"].iat[i]):
            continue
        if up[i] and c[i - 1] < e50[i - 1] and c[i] > e50[i] and r[i] > 50:
            out.append(_mk(i, 1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"], trail=p["trail_atr"]))
        elif dn[i] and c[i - 1] > e50[i - 1] and c[i] < e50[i] and r[i] < 50:
            out.append(_mk(i, -1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"], trail=p["trail_atr"]))
    return out


def vol_breakout(f: pd.DataFrame, p: dict) -> list[OrderIntent]:
    c = f["close"].to_numpy(); h = f["high"].to_numpy(); l = f["low"].to_numpy()
    a = f["atr"].to_numpy(); hh = f["hh48"].to_numpy(); ll = f["ll48"].to_numpy()
    v = f["volume"].to_numpy(); vma = f["vol_ma48"].to_numpy()
    # compression must be present on the bar BEFORE the breakout bar
    squeeze = (f["bbw_pct"].shift(1) <= p["squeeze_pct"]).to_numpy()
    rng = np.where(h - l > 0, h - l, np.nan)
    pos = (c - l) / rng
    out = []
    for i in range(len(f)):
        if np.isnan(a[i]) or np.isnan(hh[i]) or not squeeze[i] or not v[i] > p["vol_mult"] * vma[i]:
            continue
        if c[i] > hh[i] and pos[i] >= 0.75:
            out.append(_mk(i, 1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"], trail=p["trail_atr"]))
        elif c[i] < ll[i] and pos[i] <= 0.25:
            out.append(_mk(i, -1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"], trail=p["trail_atr"]))
    return out


def vwap_reversion(f: pd.DataFrame, p: dict) -> list[OrderIntent]:
    c = f["close"].to_numpy(); o = f["open"].to_numpy(); a = f["atr"].to_numpy()
    vw = f["vwap"].to_numpy(); r = f["rsi"].to_numpy()
    out = []
    for i in range(len(f)):
        if np.isnan(a[i]) or np.isnan(vw[i]) or np.isnan(r[i]):
            continue
        dev = (c[i] - vw[i]) / a[i]
        if dev < -p["dev_atr"] and r[i] < p["rsi_lo"] and c[i] > o[i]:
            out.append(_mk(i, 1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"], target=vw[i]))
        elif dev > p["dev_atr"] and r[i] > 100 - p["rsi_lo"] and c[i] < o[i]:
            out.append(_mk(i, -1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"], target=vw[i]))
    return out


def funding_momentum(f: pd.DataFrame, p: dict) -> list[OrderIntent]:
    c = f["close"].to_numpy(); a = f["atr"].to_numpy()
    roc = f["roc288"].to_numpy(); fr = f["funding_rate"].to_numpy()
    # settlement bars: the bar whose close is the settlement time (xx:45 bar before 00/08/16 UTC)
    ct = f.index + pd.Timedelta("15min")
    settle = ((ct.hour % 8 == 0) & (ct.minute == 0))
    base = p["funding_baseline"]
    out = []
    for i in np.flatnonzero(settle):
        if np.isnan(a[i]) or np.isnan(roc[i]):
            continue
        if roc[i] > 0 and fr[i] < base:
            out.append(_mk(i, 1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"]))
        elif roc[i] < 0 and fr[i] > base:
            out.append(_mk(i, -1, c[i], p["stop_atr"] * a[i], p["time_exit_bars"]))
    return out


REGISTRY = {
    "trend_pullback": trend_pullback,
    "vol_breakout": vol_breakout,
    "vwap_reversion": vwap_reversion,
    "funding_momentum": funding_momentum,
}
