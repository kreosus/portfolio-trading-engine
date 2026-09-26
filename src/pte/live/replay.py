"""Shadow replay: run the SAME bot cycle-by-cycle over recent closed bars, as if live.

At each step the bot only sees bars that had closed by that time, and the quote is the bar close
with a small spread. Fills are simulated (spec §38). This is for demonstrating and checking the
mechanics, not evidence of an edge.
"""

from __future__ import annotations

import pandas as pd

from .bot import Bot


def replay(bot: Bot, m15_all: pd.DataFrame, h1_all: pd.DataFrame, start, end=None, calendar=None,
           derivatives=None, spread_bps: float = 1.0, verbose=True) -> list[dict]:
    reports = []
    idx = m15_all.index[(m15_all.index >= start) & ((m15_all.index <= end) if end is not None else True)]
    for t in idx:
        now = t + pd.Timedelta("15min")
        m15 = m15_all[m15_all.index <= t].iloc[-1000:]
        h1 = h1_all[h1_all.index + pd.Timedelta("1h") <= now].iloc[-1200:]
        c = float(m15.close.iloc[-1])
        half = c * spread_bps * 0.5e-4
        q = {"bid": c - half, "ask": c + half, "time": now}
        r = bot.cycle(m15, h1, q, now, derivatives, calendar)
        reports.append(r)
        if verbose and (r["events"] or r.get("decision") in ("LONG", "SHORT")):
            for e in r["events"]:
                print(f"{now:%Y-%m-%d %H:%M} UTC  {e}")
    return reports
