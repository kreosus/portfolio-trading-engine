"""Trade charts ("screenshots") from the journal: candles, sweep, displacement, POI, entry, stop, TPs, exits."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

BG, FG, GRID = "#0f1115", "#d7dae0", "#23262d"
UP, DN = "#26a69a", "#ef5350"


def trade_chart(m15: pd.DataFrame, trade: dict, fills: list[dict], out_path: str, title_note: str = "SHADOW MODE"):
    sig = json.loads(trade["signal"]) if isinstance(trade["signal"], str) else trade["signal"]
    t0 = pd.Timestamp(sig["sweep"]["time"]) - pd.Timedelta("6h")
    t1 = pd.Timestamp(trade["closed_at"] or m15.index[-1]) + pd.Timedelta("4h")
    d = m15[(m15.index >= t0) & (m15.index <= t1)]
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=130)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    w = pd.Timedelta("15min").total_seconds() / 86400 * 0.7
    x = matplotlib.dates.date2num(d.index.to_pydatetime())
    for xi, (_, b) in zip(x, d.iterrows()):
        col = UP if b.close >= b.open else DN
        ax.vlines(xi, b.low, b.high, color=col, lw=0.8)
        ax.add_patch(Rectangle((xi - w / 2, min(b.open, b.close)), w, max(abs(b.close - b.open), 1e-6),
                               color=col, lw=0))
    def xt(ts):
        return matplotlib.dates.date2num(pd.Timestamp(ts).to_pydatetime())
    x_end = xt(trade["closed_at"] or d.index[-1])
    zl, zh = sig["entry_zone"]
    ax.add_patch(Rectangle((xt(sig["poi"]["time"]), zl), x_end - xt(sig["poi"]["time"]), zh - zl,
                           color="#5c6bc0", alpha=0.25, lw=0))
    ax.text(xt(sig["poi"]["time"]), zh, f" POI {sig['poi']['type']}", color="#9fa8da", fontsize=8, va="bottom")
    sw = sig["sweep"]
    ax.annotate(f"{sw['level_type']} swept", (xt(sw["time"]), sw["extreme"]),
                xytext=(0, -28 if sig["direction"] == "LONG" else 28), textcoords="offset points",
                color="#ffca28", fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color="#ffca28"))
    ax.hlines(sw["level"], xt(sw["time"]) - 0.08, xt(sw["time"]) + 0.02, color="#ffca28", lw=1, ls=":")
    dp = sig["displacement"]
    ax.annotate("displacement", (xt(dp["time"]), d.loc[pd.Timestamp(dp["time"])].close if pd.Timestamp(dp["time"]) in d.index else zh),
                xytext=(18, 18), textcoords="offset points", color="#80cbc4", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#80cbc4"))
    if sig.get("bos_choch"):
        ax.axvline(xt(sig["bos_choch"]["time"]), color="#80cbc4", lw=0.6, ls="--")
        ax.text(xt(sig["bos_choch"]["time"]), d.low.min(),
                f" {sig['bos_choch']['type']} confirmed", color="#80cbc4", fontsize=8, va="bottom")
    xo = xt(trade["opened_at"])
    lines = [("entry", trade["entry"], "#e0e0e0", "-"), ("stop", trade["stop"], DN, "--"),
             ("TP1", trade["tp1"], UP, ":"), ("TP2", trade["tp2"], UP, ":"), ("TP3", trade["tp3"], UP, ":")]
    for name, lvl, col, ls in lines:
        ax.hlines(lvl, xo, x_end, color=col, lw=1.1, ls=ls)
        ax.text(x_end, lvl, f" {name} {lvl:,.0f}", color=col, fontsize=8, va="center")
    for f in fills:
        mk = "^" if f["side"] == "BUY" else "v"
        col = "#ffffff" if f["purpose"] == "entry" else (UP if (f["realized"] or 0) > 0 else DN)
        ax.scatter(xt(f["t"]), f["price"], marker=mk, s=90, color=col, edgecolor="black", zorder=5)
        ax.annotate(f"{f['purpose']} {f['qty']:g} @ {f['price']:,.0f}", (xt(f["t"]), f["price"]),
                    xytext=(6, -14 if mk == "v" else 8), textcoords="offset points", color=col, fontsize=8)
    net = trade.get("net_pnl")
    r = trade.get("r_multiple")
    status = f"{trade['exit_reason']}, net {net:+.2f} USD ({r:+.2f}R)" if net is not None else "OPEN"
    ax.set_title(f"BTCUSDT {sig['direction']}  |  score {sig['score']:.0f}  |  {status}   "
                 f"[{title_note}: real prices, simulated fills]", color=FG, fontsize=11, loc="left")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_ylabel("USD", color=FG)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return out_path
