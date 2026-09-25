"""Strategy A — M15 liquidity/SMC.

Long setup (short mirrors it):
  1. HTF bias bullish.
  2. Bullish liquidity sweep of a swing low.
  3. Within `setup_window` bars, a bullish BOS or CHoCH closes, and the leg
     from the sweep to the break contains a bullish displacement bar.
  4. That leg leaves a bullish FVG that price has not yet traded into.
  5. Limit entry at the FVG midpoint, valid for `entry_valid_bars` bars.
  6. Stop = sweep extreme -/+ stop_buffer_atr * ATR. Target = target_r * R.

Signals are produced from closed bars only. The order becomes live on the
next bar. One setup per sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrderIntent:
    signal_idx: int        # bar on whose close the signal fired
    side: int              # +1 long, -1 short
    entry: float           # limit price; for market orders, the signal close (reference only)
    stop: float
    target: float          # use +/-inf for no target
    expires_idx: int       # last bar on which the limit may fill
    time_exit_bars: int
    entry_type: str = "limit"        # "limit" or "market" (fills at next bar open)
    trail_atr: float | None = None   # trailing stop distance in ATRs, updated on each bar close


def generate_signals(f: pd.DataFrame, cfg: dict) -> list[OrderIntent]:
    win = cfg["setup_window"]
    valid = cfg["entry_valid_bars"]
    buf = cfg["stop_buffer_atr"]
    rr = cfg["target_r"]
    tex = cfg["time_exit_bars"]

    lo = f["low"].to_numpy(); hi = f["high"].to_numpy(); close = f["close"].to_numpy()
    atr = f["atr"].to_numpy()
    htf = f["htf_trend"].to_numpy()
    sweep_up = f["sweep_up"].to_numpy(); sweep_dn = f["sweep_dn"].to_numpy()
    brk_up = (f["bos_up"] | f["choch_up"]).to_numpy()
    brk_dn = (f["bos_dn"] | f["choch_dn"]).to_numpy()
    disp_up = f["disp_up"].to_numpy(); disp_dn = f["disp_dn"].to_numpy()
    fvg_up = f["fvg_up"].to_numpy(); fvg_dn = f["fvg_dn"].to_numpy()
    fu_lo = f["fvg_up_lo"].to_numpy(); fu_hi = f["fvg_up_hi"].to_numpy()
    fd_lo = f["fvg_dn_lo"].to_numpy(); fd_hi = f["fvg_dn_hi"].to_numpy()

    out: list[OrderIntent] = []
    last_sweep_up = -1; sweep_low = np.nan
    last_sweep_dn = -1; sweep_high = np.nan
    n = len(f)
    for i in range(n):
        if sweep_up[i]:
            last_sweep_up, sweep_low = i, lo[i]
        elif last_sweep_up >= 0 and lo[i] < sweep_low:
            sweep_low = lo[i]  # sweep extends lower before the break: stop follows the true extreme
        if sweep_dn[i]:
            last_sweep_dn, sweep_high = i, hi[i]
        elif last_sweep_dn >= 0 and hi[i] > sweep_high:
            sweep_high = hi[i]

        if np.isnan(atr[i]):
            continue

        # ---- long
        s = last_sweep_up
        if brk_up[i] and s >= 0 and 0 < i - s <= win and htf[i] == 1:
            if disp_up[s:i + 1].any():
                j = _latest(fvg_up, s + 2, i)
                if j >= 0:
                    mid = (fu_lo[j] + fu_hi[j]) / 2
                    untouched = j == i or lo[j + 1:i + 1].min() > mid
                    stop = sweep_low - buf * atr[i]
                    if untouched and stop < mid < close[i]:
                        risk = mid - stop
                        out.append(OrderIntent(i, 1, mid, stop, mid + rr * risk, i + valid, tex))
            last_sweep_up = -1  # one setup per sweep

        # ---- short
        s = last_sweep_dn
        if brk_dn[i] and s >= 0 and 0 < i - s <= win and htf[i] == -1:
            if disp_dn[s:i + 1].any():
                j = _latest(fvg_dn, s + 2, i)
                if j >= 0:
                    mid = (fd_lo[j] + fd_hi[j]) / 2
                    untouched = j == i or hi[j + 1:i + 1].max() < mid
                    stop = sweep_high + buf * atr[i]
                    if untouched and close[i] < mid < stop:
                        risk = stop - mid
                        out.append(OrderIntent(i, -1, mid, stop, mid - rr * risk, i + valid, tex))
            last_sweep_dn = -1

        # expire stale sweeps
        if last_sweep_up >= 0 and i - last_sweep_up >= win:
            last_sweep_up = -1
        if last_sweep_dn >= 0 and i - last_sweep_dn >= win:
            last_sweep_dn = -1
    return out


def _latest(flags: np.ndarray, start: int, end: int) -> int:
    for j in range(end, max(start, 0) - 1, -1):
        if flags[j]:
            return j
    return -1
