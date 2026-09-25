"""Smart-money-concept detectors with exact, causal definitions.

Definitions (mirrors the Phase 1 doc):

- Swing high at bar k: high[k] strictly above the N bars on each side.
  It is only *known* at the close of bar k+N, so it is recorded at row k+N.
- Structure: bullish after a close above the last confirmed, unbroken swing
  high; bearish after a close below the last confirmed, unbroken swing low.
- BOS: a break in the direction of current structure. CHoCH: a break against it.
- Bullish liquidity sweep: low trades below the last confirmed, unswept swing
  low, but the bar closes back above it. Bearish mirrors it.
- Displacement: |body| >= k * ATR and close in the outer fraction of the range.
- Bullish FVG at bar i: low[i] > high[i-2], gap >= m * ATR[i].
  Zone = [high[i-2], low[i]]. Bearish mirrors it.
- Order block (bullish): last down-close bar before a bullish displacement bar.

Every column at row i depends only on rows <= i. tests/test_causality.py
enforces this by recomputing on truncated data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr as atr_fn


def swings(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (sh_level, sl_level) indexed by CONFIRMATION bar (k+n); NaN elsewhere."""
    size = len(high)
    sh = np.full(size, np.nan)
    sl = np.full(size, np.nan)
    for k in range(n, size - n):
        left_h, right_h = high[k - n:k], high[k + 1:k + n + 1]
        if high[k] > left_h.max() and high[k] > right_h.max():
            sh[k + n] = high[k]
        left_l, right_l = low[k - n:k], low[k + 1:k + n + 1]
        if low[k] < left_l.min() and low[k] < right_l.min():
            sl[k + n] = low[k]
    return sh, sl


def structure(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Run the structure state machine.

    Columns:
      trend        +1 bullish, -1 bearish, 0 unknown (state after bar i closes)
      bos_up/bos_dn, choch_up/choch_dn   break events on bar i
      sweep_up     bullish sweep (of a swing low) on bar i
      sweep_dn     bearish sweep (of a swing high) on bar i
      sweep_level  the swept swing level (NaN if no sweep)
      last_sh/last_sl  active (unbroken) swing levels after bar i
    """
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    sh_conf, sl_conf = swings(h, l, n)
    size = len(df)

    trend = np.zeros(size, dtype=np.int8)
    bos_up = np.zeros(size, bool); bos_dn = np.zeros(size, bool)
    choch_up = np.zeros(size, bool); choch_dn = np.zeros(size, bool)
    sweep_up = np.zeros(size, bool); sweep_dn = np.zeros(size, bool)
    sweep_level = np.full(size, np.nan)
    last_sh_arr = np.full(size, np.nan); last_sl_arr = np.full(size, np.nan)

    cur = 0
    active_sh = np.nan  # last confirmed swing high not yet broken by a close
    active_sl = np.nan
    sh_swept = False
    sl_swept = False
    for i in range(size):
        # A swing confirmed at the close of bar i becomes usable from bar i+1.
        # Evaluate breaks/sweeps against levels known before bar i opened.
        if not np.isnan(active_sh):
            if c[i] > active_sh:
                if cur == 1:
                    bos_up[i] = True
                else:
                    choch_up[i] = True
                cur = 1
                active_sh = np.nan
            elif h[i] > active_sh and not sh_swept:
                sweep_dn[i] = True
                sweep_level[i] = active_sh
                sh_swept = True
        if not np.isnan(active_sl):
            if c[i] < active_sl:
                if cur == -1:
                    bos_dn[i] = True
                else:
                    choch_dn[i] = True
                cur = -1
                active_sl = np.nan
            elif l[i] < active_sl and not sl_swept:
                sweep_up[i] = True
                sweep_level[i] = active_sl
                sl_swept = True
        # register swings confirmed at this bar's close
        if not np.isnan(sh_conf[i]):
            active_sh = sh_conf[i]
            sh_swept = False
        if not np.isnan(sl_conf[i]):
            active_sl = sl_conf[i]
            sl_swept = False
        trend[i] = cur
        last_sh_arr[i] = active_sh
        last_sl_arr[i] = active_sl

    return pd.DataFrame({
        "trend": trend, "bos_up": bos_up, "bos_dn": bos_dn,
        "choch_up": choch_up, "choch_dn": choch_dn,
        "sweep_up": sweep_up, "sweep_dn": sweep_dn, "sweep_level": sweep_level,
        "last_sh": last_sh_arr, "last_sl": last_sl_arr,
    }, index=df.index)


def displacement(df: pd.DataFrame, atr_s: pd.Series, body_atr: float, close_frac: float) -> pd.DataFrame:
    body = df["close"] - df["open"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    pos = (df["close"] - df["low"]) / rng
    big = body.abs() >= body_atr * atr_s
    up = big & (body > 0) & (pos >= 1 - close_frac)
    dn = big & (body < 0) & (pos <= close_frac)
    return pd.DataFrame({"disp_up": up.fillna(False), "disp_dn": dn.fillna(False)})


def fvg(df: pd.DataFrame, atr_s: pd.Series, min_atr: float) -> pd.DataFrame:
    h2 = df["high"].shift(2)
    l2 = df["low"].shift(2)
    up = (df["low"] > h2) & ((df["low"] - h2) >= min_atr * atr_s)
    dn = (df["high"] < l2) & ((l2 - df["high"]) >= min_atr * atr_s)
    return pd.DataFrame({
        "fvg_up": up.fillna(False),
        "fvg_up_lo": h2.where(up), "fvg_up_hi": df["low"].where(up),
        "fvg_dn": dn.fillna(False),
        "fvg_dn_lo": df["high"].where(dn), "fvg_dn_hi": l2.where(dn),
    })


def order_blocks(df: pd.DataFrame, disp: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """On each displacement bar, the zone of the last opposite-colour bar before it."""
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    h = df["high"].to_numpy(); l = df["low"].to_numpy()
    du = disp["disp_up"].to_numpy(); dd = disp["disp_dn"].to_numpy()
    size = len(df)
    ob_up_lo = np.full(size, np.nan); ob_up_hi = np.full(size, np.nan)
    ob_dn_lo = np.full(size, np.nan); ob_dn_hi = np.full(size, np.nan)
    for i in range(size):
        if du[i] or dd[i]:
            for j in range(i - 1, max(-1, i - 1 - lookback), -1):
                if du[i] and c[j] < o[j]:
                    ob_up_lo[i], ob_up_hi[i] = l[j], h[j]
                    break
                if dd[i] and c[j] > o[j]:
                    ob_dn_lo[i], ob_dn_hi[i] = l[j], h[j]
                    break
    return pd.DataFrame({"ob_up_lo": ob_up_lo, "ob_up_hi": ob_up_hi,
                         "ob_dn_lo": ob_dn_lo, "ob_dn_hi": ob_dn_hi}, index=df.index)


def htf_bias(df: pd.DataFrame, rule: str, n: int) -> pd.Series:
    """Higher-timeframe structure trend mapped onto base bars without look-ahead.

    A HTF bar covering [t, t+rule) is known only once the base bar ending at
    t+rule has closed. Incomplete HTF bars (e.g. at a data gap) are dropped.
    """
    base = pd.Timedelta(df.index.freq or pd.infer_freq(df.index[:100]) or "15min")
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = df["close"].resample(rule, label="left", closed="left").count()
    need = int(pd.Timedelta(rule) / base)
    agg = agg[counts == need].dropna()
    st = structure(agg, n)
    avail = pd.DataFrame({"avail": agg.index + pd.Timedelta(rule), "htf_trend": st["trend"].to_numpy()})
    base_close = pd.DataFrame({"close_time": df.index + base})
    merged = pd.merge_asof(base_close, avail, left_on="close_time", right_on="avail", direction="backward")
    return pd.Series(merged["htf_trend"].fillna(0).astype(int).to_numpy(), index=df.index, name="htf_trend")


def build_features(df: pd.DataFrame, smc_cfg: dict) -> pd.DataFrame:
    a = atr_fn(df, smc_cfg["atr_period"])
    st = structure(df, smc_cfg["swing_n"])
    disp = displacement(df, a, smc_cfg["displacement_body_atr"], smc_cfg["displacement_close_frac"])
    gaps = fvg(df, a, smc_cfg["fvg_min_atr"])
    obs = order_blocks(df, disp)
    htf = htf_bias(df, smc_cfg["htf_rule"], smc_cfg["htf_swing_n"])
    return pd.concat([df, a, st, disp, gaps, obs, htf], axis=1)
