"""Analytical engines (spec §7-§18). Each returns a plain dict; none of them places orders or decides a trade.

Inputs are CLOSED bars only. Every function looks at data up to the last bar it is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.indicators import atr as atr_fn, ema, rsi, session_vwap
from ..features.smc import fvg as fvg_fn, structure as structure_fn, swings


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    base = df.index[1] - df.index[0]
    need = int(pd.Timedelta(rule) / base)
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    n = df["close"].resample(rule, label="left", closed="left").count()
    return agg[n == need].dropna()


# ----------------------------------------------------------------------------- structure (§7)

def structure_engine(df: pd.DataFrame, n: int) -> dict:
    st = structure_fn(df, n)
    last = st.iloc[-1]
    trend = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}[int(last.trend)]
    ev = st[st.bos_up | st.bos_dn | st.choch_up | st.choch_dn]
    last_bos = st[st.bos_up | st.bos_dn].tail(1)
    last_choch = st[st.choch_up | st.choch_dn].tail(1)
    # strength: consecutive breaks in the current direction among the last events
    strength = 0
    for _, r in ev.iloc[::-1].iterrows():
        up = bool(r.bos_up or r.choch_up)
        if (up and last.trend == 1) or (not up and last.trend == -1):
            strength += 1
            if r.choch_up or r.choch_dn:
                break
        else:
            break
    sh, sl = swings(df.high.to_numpy(), df.low.to_numpy(), n)
    sh_s = pd.Series(sh, index=df.index).dropna()
    sl_s = pd.Series(sl, index=df.index).dropna()
    hi = float(sh_s.iloc[-1]) if len(sh_s) else float(df.high.max())
    lo = float(sl_s.iloc[-1]) if len(sl_s) else float(df.low.min())
    c = float(df.close.iloc[-1])
    rng_lo, rng_hi = min(lo, hi), max(lo, hi)
    pos = (c - rng_lo) / (rng_hi - rng_lo) if rng_hi > rng_lo else 0.5
    return {
        "direction": trend, "strength": strength,
        "last_bos": None if last_bos.empty else {"time": str(last_bos.index[-1]),
                                                   "dir": "UP" if last_bos.bos_up.iloc[-1] else "DOWN"},
        "last_choch": None if last_choch.empty else {"time": str(last_choch.index[-1]),
                                                       "dir": "UP" if last_choch.choch_up.iloc[-1] else "DOWN"},
        "major_swing_high": hi, "major_swing_low": lo,
        "premium_discount": "PREMIUM" if pos > 0.5 else "DISCOUNT", "range_position": round(pos, 3),
        "_frame": st,
    }


def htf_bias(h4: dict, h1: dict) -> str:
    if h4["direction"] == h1["direction"] and h4["direction"] != "NEUTRAL":
        return h4["direction"]
    if h4["direction"] == "NEUTRAL" and h1["direction"] != "NEUTRAL":
        return h1["direction"]
    return "MIXED" if h4["direction"] != "NEUTRAL" and h1["direction"] != "NEUTRAL" else "NEUTRAL"


# ----------------------------------------------------------------------------- liquidity (§8)

def _session_window(ts: pd.Timestamp, start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = ts.floor("D")
    s = day + pd.Timedelta(start + ":00")
    e = day + pd.Timedelta(end + ":00")
    return s, e


def liquidity_levels(m15: pd.DataFrame, h1: pd.DataFrame, n: int, cfg: dict) -> list[dict]:
    """Where liquidity rests, as of the last closed M15 bar."""
    t = m15.index[-1]
    a = float(atr_fn(m15, 14).iloc[-1])
    lv = []
    day = t.floor("D")
    prev_day = m15[(m15.index >= day - pd.Timedelta("1D")) & (m15.index < day)]
    if len(prev_day):
        lv += [{"type": "PDH", "side": "BUY", "level": float(prev_day.high.max()), "weight": 1.0},
               {"type": "PDL", "side": "SELL", "level": float(prev_day.low.min()), "weight": 1.0}]
    wk = (t - pd.Timedelta(days=t.weekday())).floor("D")
    prev_wk = h1[(h1.index >= wk - pd.Timedelta("7D")) & (h1.index < wk)]
    if len(prev_wk):
        lv += [{"type": "PWH", "side": "BUY", "level": float(prev_wk.high.max()), "weight": 1.3},
               {"type": "PWL", "side": "SELL", "level": float(prev_wk.low.min()), "weight": 1.3}]
    for name, (st, en) in cfg["sessions_utc"].items():
        s, e = _session_window(t, st, en)
        if e > t:                         # use the most recent COMPLETED session
            s, e = s - pd.Timedelta("1D"), e - pd.Timedelta("1D")
        seg = m15[(m15.index >= s) & (m15.index < e)]
        if len(seg):
            lv += [{"type": f"{name.upper()}_HIGH", "side": "BUY", "level": float(seg.high.max()), "weight": 0.8},
                   {"type": f"{name.upper()}_LOW", "side": "SELL", "level": float(seg.low.min()), "weight": 0.8}]
    sh, sl = swings(m15.high.to_numpy(), m15.low.to_numpy(), n)
    highs = pd.Series(sh, index=m15.index).dropna().tail(12)
    lows = pd.Series(sl, index=m15.index).dropna().tail(12)
    tol = cfg["liquidity"]["eq_tolerance_atr"] * a
    for series, side, typ in ((highs, "BUY", "EQH"), (lows, "SELL", "EQL")):
        vals = series.to_numpy()
        used = set()
        for i in range(len(vals)):
            grp = [j for j in range(len(vals)) if abs(vals[j] - vals[i]) <= tol]
            if len(grp) >= 2 and i not in used:
                used.update(grp)
                lv.append({"type": typ, "side": side, "level": float(np.mean(vals[grp])), "weight": 1.2,
                           "touches": len(grp)})
        for v in vals[-4:]:
            lv.append({"type": "SWING_HIGH" if side == "BUY" else "SWING_LOW", "side": side,
                       "level": float(v), "weight": 0.6})
    # H1 swing points over roughly the last two weeks: the obvious resting liquidity (spec §8)
    h1r = h1.iloc[-336:]
    hh, hl = swings(h1r.high.to_numpy(), h1r.low.to_numpy(), n)
    for v in pd.Series(hh).dropna().tail(10):
        lv.append({"type": "H1_SWING_HIGH", "side": "BUY", "level": float(v), "weight": 1.1})
    for v in pd.Series(hl).dropna().tail(10):
        lv.append({"type": "H1_SWING_LOW", "side": "SELL", "level": float(v), "weight": 1.1})
    return lv


def detect_sweeps(m15: pd.DataFrame, levels: list[dict], cfg: dict) -> list[dict]:
    """A sweep = trades beyond a level by >= min depth and CLOSES back inside, within the lookback.
    A mere touch is not a sweep (spec §8). Levels must pre-date the sweeping bar."""
    a = atr_fn(m15, 14)
    look = cfg["liquidity"]["sweep_lookback_bars"]
    min_depth = cfg["liquidity"]["min_sweep_depth_atr"]
    out = []
    recent = m15.iloc[-look:]
    for i, (ts, bar) in enumerate(recent.iterrows()):
        at = float(a.loc[ts])
        for L in levels:
            if L["side"] == "SELL":        # sell-side liquidity below lows -> bullish sweep
                depth = L["level"] - bar.low
                if depth >= min_depth * at and bar.close > L["level"]:
                    out.append({"direction": "SELL_SIDE", "bias": "LONG", "time": str(ts), "bar": ts,
                                "level_type": L["type"], "level": L["level"], "extreme": float(bar.low),
                                "strength": round(min(depth / at, 3.0) * L["weight"], 2)})
            else:
                depth = bar.high - L["level"]
                if depth >= min_depth * at and bar.close < L["level"]:
                    out.append({"direction": "BUY_SIDE", "bias": "SHORT", "time": str(ts), "bar": ts,
                                "level_type": L["type"], "level": L["level"], "extreme": float(bar.high),
                                "strength": round(min(depth / at, 3.0) * L["weight"], 2)})
    return out


# ----------------------------------------------------------------------------- displacement (§11)

def displacement_after(m15: pd.DataFrame, start_ts, bias: str, st_frame: pd.DataFrame, cfg: dict) -> dict | None:
    """First qualifying displacement bar after the sweep, with the structure break it produced."""
    d = cfg["displacement"]
    a = atr_fn(m15, 14)
    rng = (m15.high - m15.low)
    med_rng = rng.rolling(20).median().shift(1)
    med_vol = m15.volume.rolling(20).median().shift(1)
    seg = m15[m15.index >= start_ts]
    for ts, b in seg.iterrows():
        body = b.close - b.open
        up = bias == "LONG"
        if (up and body <= 0) or (not up and body >= 0):
            continue
        r = b.high - b.low
        pos = (b.close - b.low) / r if r > 0 else 0.5
        big = abs(body) >= d["body_atr"] * a.loc[ts]
        closes_strong = pos >= 1 - d["close_frac"] if up else pos <= d["close_frac"]
        expansion = r >= d["range_expansion"] * med_rng.loc[ts] if not np.isnan(med_rng.loc[ts]) else False
        vol = b.volume >= d["volume_mult"] * med_vol.loc[ts] if not np.isnan(med_vol.loc[ts]) else False
        i = m15.index.get_loc(ts)
        k, consec = i, 0
        while k >= 0 and ((m15.close.iloc[k] > m15.open.iloc[k]) if up else (m15.close.iloc[k] < m15.open.iloc[k])):
            consec += 1
            k -= 1
        if not (big and closes_strong and (expansion or vol)):
            continue
        after = st_frame[st_frame.index >= ts]
        brk = after[(after.bos_up | after.choch_up) if up else (after.bos_dn | after.choch_dn)].head(1)
        return {"time": str(ts), "bar": ts, "body_atr": round(abs(body) / a.loc[ts], 2),
                "range_expansion": bool(expansion), "volume_confirm": bool(vol), "consecutive": consec,
                "structure_break": None if brk.empty else {
                    "time": str(brk.index[0]),
                    "type": ("CHoCH" if (brk.choch_up.iloc[0] if up else brk.choch_dn.iloc[0]) else "BOS")}}
    return None


# ----------------------------------------------------------------------------- POI (§12)

def poi_engine(m15: pd.DataFrame, disp: dict, bias: str, cfg: dict, dealing_range: tuple) -> dict | None:
    a = atr_fn(m15, 14)
    g = fvg_fn(m15, a, cfg["poi"]["fvg_min_atr"])
    up = bias == "LONG"
    leg = m15[m15.index >= disp["bar"] - pd.Timedelta("30min")]
    col = "fvg_up" if up else "fvg_dn"
    cand = g.loc[leg.index][g.loc[leg.index, col]]
    last_i = len(m15) - 1
    best = None
    for ts in cand.index[::-1]:
        lo = float(g.loc[ts, "fvg_up_lo" if up else "fvg_dn_lo"])
        hi = float(g.loc[ts, "fvg_up_hi" if up else "fvg_dn_hi"])
        mid = (lo + hi) / 2
        age = last_i - m15.index.get_loc(ts)
        after = m15[m15.index > ts]
        mitigated = bool((after.low <= mid).any()) if up else bool((after.high >= mid).any())
        if age > cfg["poi"]["max_age_bars"] or mitigated:
            continue
        best = {"type": "FVG", "time": str(ts), "low": lo, "high": hi, "mid": mid, "age_bars": int(age),
                "size_atr": round((hi - lo) / float(a.loc[ts]), 2), "mitigated": False}
        break
    # order block: last opposite-colour candle before the displacement bar
    i = m15.index.get_loc(disp["bar"])
    ob = None
    for j in range(i - 1, max(-1, i - 11), -1):
        b = m15.iloc[j]
        if (up and b.close < b.open) or (not up and b.close > b.open):
            after = m15.iloc[j + 1:]
            mitigated = bool((after.iloc[1:].low <= b.high).any()) if up else bool((after.iloc[1:].high >= b.low).any())
            ob = {"type": "OB", "time": str(m15.index[j]), "low": float(b.low), "high": float(b.high),
                  "mid": float((b.low + b.high) / 2), "mitigated": mitigated}
            break
    if best is None and (ob is None or ob["mitigated"]):
        return None
    zone = best or ob
    overlap = bool(best and ob and best["low"] <= ob["high"] and ob["low"] <= best["high"])
    lo_r, hi_r = dealing_range
    eq = (lo_r + hi_r) / 2
    in_discount = zone["mid"] < eq if up else zone["mid"] > eq
    return {"zone": zone, "fvg": best, "ob": ob, "fvg_ob_overlap": overlap,
            "location": ("DISCOUNT" if up else "PREMIUM") if in_discount else ("PREMIUM" if up else "DISCOUNT"),
            "favourable_location": bool(in_discount)}


# ----------------------------------------------------------------------------- momentum (§13)

def momentum_engine(m15: pd.DataFrame, cfg: dict) -> dict:
    m = cfg["momentum"]
    c = m15.close
    r = float(rsi(c, m["rsi_period"]).iloc[-1])
    ef, es = float(ema(c, m["ema_fast"]).iloc[-1]), float(ema(c, m["ema_slow"]).iloc[-1])
    vw = float(session_vwap(m15).iloc[-1])
    vol_ratio = float(m15.volume.iloc[-8:].mean() / m15.volume.iloc[-100:-8].median())
    return {"rsi": round(r, 1), "ema_fast": round(ef, 2), "ema_slow": round(es, 2), "ema_state":
            "BULLISH" if ef > es else "BEARISH", "vwap": round(vw, 2), "close": float(c.iloc[-1]),
            "above_vwap": bool(c.iloc[-1] > vw), "volume_ratio": round(vol_ratio, 2)}


# ----------------------------------------------------------------------------- context (§14, §15, §18)

def session_of(ts: pd.Timestamp, cfg: dict) -> str:
    hits = []
    for name, (s, e) in cfg["sessions_utc"].items():
        a, b = _session_window(ts, s, e)
        if a <= ts < b:
            hits.append(name.upper())
    return "+".join(hits) if hits else "OFF_HOURS"


def regime_engine(h4: pd.DataFrame, h4_struct: dict) -> dict:
    a = atr_fn(h4, 14)
    pct = float(a.rank(pct=True).iloc[-1])
    slope = float(a.iloc[-1] / a.iloc[-6] - 1) if len(a) > 6 and a.iloc[-6] > 0 else 0.0
    trend = h4_struct["direction"]
    labels = []
    if trend == "BULLISH" and h4_struct["strength"] >= 2:
        labels.append("TRENDING_BULL")
    elif trend == "BEARISH" and h4_struct["strength"] >= 2:
        labels.append("TRENDING_BEAR")
    else:
        labels.append("RANGING")
    labels.append("HIGH_VOLATILITY" if pct > 0.8 else "LOW_VOLATILITY" if pct < 0.2 else "NORMAL_VOLATILITY")
    if slope > 0.25:
        labels.append("VOLATILITY_EXPANSION")
    elif slope < -0.25:
        labels.append("VOLATILITY_COMPRESSION")
    return {"regime": labels[0], "labels": labels, "atr_percentile": round(pct, 2)}


# ----------------------------------------------------------------------------- news (§16, §17, §41)

def news_state(events: list | None, now: pd.Timestamp, cfg: dict) -> dict:
    n = cfg["news"]
    if events is None:
        return {"state": "NEWS_DATA_UNAVAILABLE", "blocking": n["unavailable_policy"] == "block",
                "penalty": 0 if n["unavailable_policy"] == "block" else 10, "event": None}
    high = []
    for e in events:
        if e.get("country") != "USD" or e.get("impact") != "High":
            continue
        t = pd.Timestamp(e["date"]).tz_convert("UTC")
        high.append((t, e))
    state, blocking, penalty, ev = "NORMAL", False, 0, None
    for t, e in sorted(high, key=lambda x: x[0]):
        until = (t - now).total_seconds() / 60
        since = -until
        if 0 <= until <= n["hard_block_minutes"] or 0 <= since <= n["post_lock_minutes"]:
            state, blocking, ev = ("EVENT_LOCK" if until >= 0 or since < 5 else "POST_EVENT_VOLATILITY"), True, e
            break
        if 0 <= until <= n["restrict_minutes"]:
            state, blocking, ev = "PRE_EVENT", True, e
            break
        if 0 <= until <= n["pre_event_minutes"]:
            state, penalty, ev = "PRE_EVENT", 10, e
        elif n["post_lock_minutes"] < since <= n["reassess_minutes"] and state == "NORMAL":
            state, penalty, ev = "REASSESSMENT", 10, e
    out = {"state": state, "blocking": blocking, "penalty": penalty, "event": None}
    if ev:
        t = pd.Timestamp(ev["date"]).tz_convert("UTC")
        out["event"] = {"title": ev.get("title"), "time": str(t), "minutes_until": round((t - now).total_seconds() / 60)}
    return out
