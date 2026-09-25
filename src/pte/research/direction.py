"""Direction-filter study (exploration on 2020-2025 development data).

Hypothesis, stated before running: the candidates' 2020-2025 profits leaned on
longs in a bull market. Only trading WITH the 200-day trend (longs above the
200-day average, shorts below) should keep most of the edge and lose less when
the market turns down.

For each mode the random-entry baseline gets the same filter, so the question
is whether the signal beats random entries taken in the same trend condition.
Results here are exploratory: 2020-2025 has been used many times.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..backtest.engine import SleeveSpec, run_portfolio
from ..backtest.metrics import summarize
from ..bots import build_features_all
from ..data.store import resample_bars
from ..strategies.library import REGISTRY, apply_direction_filter
from .robustness import _research_cfg

MODES = ("none", "trend", "long_only")


def _run(feats, funding, rc, name, intents, start):
    return run_portfolio(feats, [SleeveSpec(name, intents, 1.0)], funding, rc["risk"],
                         CostModel.from_config(rc["costs"]), start).sleeves[name]


def _random_like(real, feats, params, start, mode, rng):
    out = {}
    for s, lst in real.items():
        f = feats[s]
        live = [x for x in lst if f.index[x.signal_idx] >= start]
        if not live:
            out[s] = []
            continue
        c = f["close"].to_numpy(); a = f["atr"].to_numpy(); sma = f["sma200d"].to_numpy()
        ok = (f.index >= start) & ~np.isnan(a)
        n_long = sum(x.side == 1 for x in live); n_short = len(live) - n_long
        picks = []
        for side, n in ((1, n_long), (-1, n_short)):
            if n == 0:
                continue
            cond = ok.copy()
            if mode in ("trend", "long_only"):
                cond &= (c > sma) if side == 1 else (c < sma)
            pool = np.flatnonzero(cond)
            if len(pool) == 0:
                continue
            for i in rng.choice(pool, size=min(n, len(pool)), replace=False):
                picks.append((int(i), side))
        tmpl = live[0]
        out[s] = [replace(tmpl, signal_idx=i, side=side, entry=c[i],
                          stop=c[i] - side * params["stop_atr"] * a[i], target=side * np.inf,
                          expires_idx=i + 1) for i, side in sorted(picks)]
    return out


def direction_study(bars15: dict, funding: dict, cfg: dict, name: str, tf: str, start,
                    n_seeds: int = 30, seed: int = 0) -> pd.DataFrame:
    rc = _research_cfg(cfg, tf)
    bars = {s: resample_bars(df, tf) for s, df in bars15.items()}
    feats = build_features_all(bars, funding, rc)
    params = {k: v for k, v in rc["bots"]["sleeves"][name].items() if k != "enabled"}
    base = {s: REGISTRY[name](f, params) for s, f in feats.items()}
    rows = []
    for mode in MODES:
        intents = {s: apply_direction_filter(lst, feats[s], mode) for s, lst in base.items()}
        r = _run(feats, funding, rc, name, intents, start)
        m = summarize(r.trades, r.equity, r.initial_equity)
        t = r.trades
        eq = r.equity.resample("1D").last()
        yearly = eq.groupby(eq.index.year).last() / eq.groupby(eq.index.year).first() - 1
        rng = np.random.default_rng(seed)
        rand = [_run(feats, funding, rc, name, _random_like(intents, feats, params, start, mode, rng), start)
                .trades.r.mean() for _ in range(n_seeds)]
        rand = np.array([x for x in rand if not np.isnan(x)])
        down = t[pd.to_datetime(t.exit_time).dt.year == 2022] if len(t) else t
        rows.append({
            "mode": mode, "trades": m["trades"], "return": m["total_return"], "sharpe": m["sharpe"],
            "pf": m["profit_factor"], "max_dd": m["max_drawdown"], "avg_r": m["avg_r"],
            "long_share": float((t.side == 1).mean()) if len(t) else 0.0,
            "r_2022_bear": float(down.r.sum()) if len(down) else 0.0,
            "profitable_years": f"{int((yearly > 0).sum())}/{len(yearly)}",
            "random_avg_r": float(np.median(rand)) if len(rand) else np.nan,
            "random_ge_real": float((rand >= m["avg_r"]).mean()) if len(rand) else np.nan,
        })
    return pd.DataFrame(rows).set_index("mode")
