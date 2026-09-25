"""Robustness diagnostics for a sub-bot on development data.

- Parameter neighbourhood: each numeric parameter moved x0.75 and x1.25, one at a
  time. A real effect should form a plateau; a lucky setting shows up as a spike.
- Per-year returns, per-symbol and long/short splits.
- Bootstrap of per-trade R: 95% interval for the mean, and the share of resamples
  with mean <= 0 (a rough one-sided p-value, NOT corrected for how many
  configurations were tried).
- Monte Carlo max drawdown in R from reshuffled trade order.

Measured in research mode (drawdown halt and loss-streak pause off) so the whole
window is used. This is diagnosis, not tuning: nothing here changes the config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..backtest.engine import SleeveSpec, run_portfolio
from ..backtest.metrics import summarize
from ..bots import build_features_all
from ..config import with_overrides
from ..data.store import HTF_FOR, resample_bars
from ..strategies.library import REGISTRY

INT_PARAMS = {"time_exit_bars"}


def _research_cfg(cfg: dict, tf: str) -> dict:
    return with_overrides(cfg, {"smc.htf_rule": HTF_FOR[tf], "risk.drawdown_halt": 1.0,
                                "risk.max_consecutive_losses": 10**9})


def _run(feats, funding, cfg, name, params, start):
    intents = {s: REGISTRY[name](f, params) for s, f in feats.items()}
    res = run_portfolio(feats, [SleeveSpec(name, intents, 1.0)], funding, cfg["risk"],
                        CostModel.from_config(cfg["costs"]), start)
    return res.sleeves[name]


def robustness(bars15: dict, funding: dict, cfg: dict, name: str, tf: str, start,
               n_boot: int = 5000, seed: int = 0) -> dict:
    rc = _research_cfg(cfg, tf)
    bars = {s: resample_bars(df, tf) for s, df in bars15.items()}
    feats = build_features_all(bars, funding, rc)
    base = {k: v for k, v in rc["bots"]["sleeves"][name].items() if k != "enabled"}

    # 1) neighbourhood
    rows = []
    variants = [("base", None, None)]
    for k, v in base.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            for mult in (0.75, 1.25):
                variants.append((f"{k} x{mult}", k, mult))
    base_res = None
    for label, k, mult in variants:
        p = dict(base)
        if k is not None:
            nv = p[k] * mult
            p[k] = int(round(nv)) if k in INT_PARAMS else nv
        r = _run(feats, funding, rc, name, p, start)
        m = summarize(r.trades, r.equity, r.initial_equity)
        rows.append({"variant": label, "trades": m["trades"], "return": m["total_return"],
                     "sharpe": m["sharpe"], "pf": m["profit_factor"], "avg_r": m["avg_r"]})
        if k is None:
            base_res = r
    neigh = pd.DataFrame(rows).set_index("variant")

    t = base_res.trades.copy()
    t["year"] = pd.to_datetime(t.exit_time).dt.year
    eq = base_res.equity.resample("1D").last()
    yearly = eq.groupby(eq.index.year).last() / eq.groupby(eq.index.year).first() - 1
    splits = pd.concat({
        "symbol": t.groupby("symbol").r.agg(["count", "mean", "sum"]),
        "side": t.groupby(t.side.map({1: "long", -1: "short"})).r.agg(["count", "mean", "sum"]),
    })

    # 3) bootstrap + Monte Carlo on per-trade R
    rng = np.random.default_rng(seed)
    r_arr = t.r.to_numpy()
    boots = rng.choice(r_arr, size=(n_boot, len(r_arr)), replace=True).mean(axis=1)
    mc_dd = []
    for _ in range(1000):
        path = np.cumsum(rng.permutation(r_arr))
        mc_dd.append(float(np.max(np.maximum.accumulate(np.concatenate([[0], path]))[1:] - path)))
    stats = {
        "trades": len(r_arr), "mean_r": float(r_arr.mean()),
        "mean_r_ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "p_mean_le_0": float((boots <= 0).mean()),
        "mc_max_dd_r_median": float(np.median(mc_dd)), "mc_max_dd_r_p95": float(np.percentile(mc_dd, 95)),
        "neighbour_share_profitable": float((neigh.drop("base")["return"] > 0).mean()),
    }
    return {"neighbourhood": neigh, "yearly": yearly, "splits": splits, "stats": stats}


def random_entry_baseline(bars15: dict, funding: dict, cfg: dict, name: str, tf: str, start,
                          n_seeds: int = 40, seed: int = 0) -> dict:
    """Same exits, same trade count per symbol and side, but entry bars chosen at random.

    If random entries earn about as much per trade, the entry signal adds nothing
    beyond the market's drift over the period (e.g. being long in a bull market).
    """
    from dataclasses import replace
    rc = _research_cfg(cfg, tf)
    bars = {s: resample_bars(df, tf) for s, df in bars15.items()}
    feats = build_features_all(bars, funding, rc)
    params = {k: v for k, v in rc["bots"]["sleeves"][name].items() if k != "enabled"}
    real = {s: REGISTRY[name](f, params) for s, f in feats.items()}
    real_res = run_portfolio(feats, [SleeveSpec(name, real, 1.0)], funding, rc["risk"],
                             CostModel.from_config(rc["costs"]), start).sleeves[name]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_seeds):
        rand = {}
        for s, lst in real.items():
            f = feats[s]
            ok = np.flatnonzero((f.index >= start) & f["atr"].notna().to_numpy())
            picks = np.sort(rng.choice(ok, size=len([x for x in lst if f.index[x.signal_idx] >= start]),
                                       replace=False))
            sides = [x.side for x in lst if f.index[x.signal_idx] >= start]
            rng.shuffle(sides)
            c = f["close"].to_numpy(); a = f["atr"].to_numpy()
            tmpl = lst[0]
            out = []
            for i, side in zip(picks, sides):
                stop_dist = abs(tmpl.entry - tmpl.stop) / (abs(tmpl.entry - tmpl.stop) or 1) * params["stop_atr"] * a[i]
                tgt = side * np.inf
                out.append(replace(tmpl, signal_idx=int(i), side=side, entry=c[i], stop=c[i] - side * stop_dist,
                                   target=tgt, expires_idx=int(i) + 1))
            rand[s] = out
        r = run_portfolio(feats, [SleeveSpec(name, rand, 1.0)], funding, rc["risk"],
                          CostModel.from_config(rc["costs"]), start).sleeves[name]
        means.append(float(r.trades.r.mean()))
    means = np.array(means)
    real_mean = float(real_res.trades.r.mean())
    return {"real_mean_r": real_mean, "random_mean_r_median": float(np.median(means)),
            "random_mean_r_p5_p95": (float(np.percentile(means, 5)), float(np.percentile(means, 95))),
            "share_random_ge_real": float((means >= real_mean).mean()), "n_seeds": n_seeds}
